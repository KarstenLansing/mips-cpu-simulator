"""
MIPS CPU Simulator 
============================================================
A unified, pure-Python simulation of a 32-bit MIPS processor.

  Lab 1 — Clock, MUX, ALU primitives
  Lab 2 — Instruction decoder + R-type ALU execution
  Lab 3 — Single-cycle CPU (R-type + I-type + memory + branch)
  Lab 5 — 5-stage pipelined CPU with data forwarding
  Lab 6 — Branch prediction (1-bit, 2-bit, BHT)
  Lab 7 — 4-way set-associative cache
  Lab 8 — Two-level page table walker
  Lab 9 — Reorder buffer (out-of-order commit tracking)

Run:
    python3 mips_cpu_sim.py                     # runs built-in demo program
    python3 mips_cpu_sim.py --file program.hex  # loads hex instructions

Author: Karsten Lansing
"""

from __future__ import annotations
import argparse
import struct
import sys
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a *bits*-wide integer to a full Python int."""
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value

def to_unsigned_32(value: int) -> int:
    """Clamp to unsigned 32-bit."""
    return value & 0xFFFFFFFF

def to_signed_32(value: int) -> int:
    """Interpret an unsigned 32-bit value as signed."""
    v = value & 0xFFFFFFFF
    if v & 0x80000000:
        return v - 0x100000000
    return v

# ─────────────────────────────────────────────────────────────────────────────
# Lab 1 — Clock (cycle counter)
# ─────────────────────────────────────────────────────────────────────────────

class Clock:
    """24-hour clock that counts HH:MM:SS, one second per tick (Lab 1)."""

    def __init__(self):
        self.seconds = 0
        self.minutes = 0
        self.hours = 0

    def tick(self):
        self.seconds += 1
        if self.seconds == 60:
            self.seconds = 0
            self.minutes += 1
            if self.minutes == 60:
                self.minutes = 0
                self.hours += 1
                if self.hours == 24:
                    self.hours = 0

    def __repr__(self):
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}"

# ─────────────────────────────────────────────────────────────────────────────
# Lab 1 — ALU (1-bit simplified + 32-bit full)
# ─────────────────────────────────────────────────────────────────────────────

class ALU:
    """
    32-bit MIPS ALU supporting the operations built up across Labs 1-3:
        ADD, SUB, AND, OR, XOR, SLT, SLL, SRL, SRA, LUI, NOR
    Returns (result, zero_flag).
    """

    # ALU control codes (matches Lab 5 encoding)
    ADD  = 0x0
    AND  = 0x1
    OR   = 0x2
    SLT  = 0x3
    SUB  = 0x4
    LUI  = 0x5
    XOR  = 0x6
    SLL  = 0x7
    SRL  = 0x8
    SRA  = 0x9
    NOR  = 0xA

    @staticmethod
    def execute(a: int, b: int, op: int, shamt: int = 0) -> Tuple[int, bool]:
        a, b = to_unsigned_32(a), to_unsigned_32(b)
        if op == ALU.ADD:
            result = a + b
        elif op == ALU.SUB:
            result = a - b
        elif op == ALU.AND:
            result = a & b
        elif op == ALU.OR:
            result = a | b
        elif op == ALU.XOR:
            result = a ^ b
        elif op == ALU.NOR:
            result = ~(a | b)
        elif op == ALU.SLT:
            result = 1 if to_signed_32(a) < to_signed_32(b) else 0
        elif op == ALU.SLL:
            result = b << (shamt & 0x1F)
        elif op == ALU.SRL:
            result = b >> (shamt & 0x1F)
        elif op == ALU.SRA:
            result = to_signed_32(b) >> (shamt & 0x1F)
        elif op == ALU.LUI:
            result = (b & 0xFFFF) << 16
        else:
            result = 0
        result = to_unsigned_32(result)
        return result, (result == 0)

# ─────────────────────────────────────────────────────────────────────────────
# Lab 2 — Instruction Decoder
# ─────────────────────────────────────────────────────────────────────────────

class DecodedInstruction:
    """Decoded MIPS instruction fields (Lab 2)."""

    __slots__ = ("raw", "op", "rs", "rt", "rd", "shamt", "funct",
                 "imm", "imm_sign", "imm_zero", "addr")

    def __init__(self, raw: int):
        self.raw       = to_unsigned_32(raw)
        self.op        = (self.raw >> 26) & 0x3F
        self.rs        = (self.raw >> 21) & 0x1F
        self.rt        = (self.raw >> 16) & 0x1F
        self.rd        = (self.raw >> 11) & 0x1F
        self.shamt     = (self.raw >>  6) & 0x1F
        self.funct     = self.raw & 0x3F
        self.imm       = self.raw & 0xFFFF
        self.imm_sign  = sign_extend(self.imm, 16)
        self.imm_zero  = self.imm
        self.addr      = self.raw & 0x03FFFFFF

    @property
    def is_r_type(self) -> bool:
        return self.op == 0

    @property
    def is_nop(self) -> bool:
        return self.raw == 0

    def __repr__(self):
        return (f"Instr(op={self.op:#04x} rs={self.rs} rt={self.rt} "
                f"rd={self.rd} sh={self.shamt} fn={self.funct:#04x} "
                f"imm={self.imm_sign})")

# ─────────────────────────────────────────────────────────────────────────────
# Lab 3 — Control Unit
# ─────────────────────────────────────────────────────────────────────────────

class ControlSignals:
    """Control signals generated by the controller (Labs 3 & 5)."""

    __slots__ = ("reg_dst", "branch", "reg_write", "alu_src",
                 "mem_write", "mem_to_reg", "alu_op", "jump")

    def __init__(self):
        self.reg_dst   = 0
        self.branch    = 0
        self.reg_write = 0
        self.alu_src   = 0   # 0=reg, 1=sign-ext imm, 2=zero-ext imm
        self.mem_write = 0
        self.mem_to_reg = 0
        self.alu_op    = ALU.ADD
        self.jump      = 0

    @staticmethod
    def decode(inst: DecodedInstruction) -> "ControlSignals":
        c = ControlSignals()
        op, fn = inst.op, inst.funct
        if op == 0x00:  # R-type
            c.reg_dst = 1
            c.reg_write = 1
            funct_map = {
                0x20: ALU.ADD, 0x22: ALU.SUB, 0x24: ALU.AND,
                0x25: ALU.OR,  0x26: ALU.XOR, 0x27: ALU.NOR,
                0x2A: ALU.SLT, 0x00: ALU.SLL, 0x02: ALU.SRL,
                0x03: ALU.SRA,
            }
            c.alu_op = funct_map.get(fn, ALU.ADD)
        elif op == 0x08:  # ADDI
            c.reg_write = 1; c.alu_src = 1; c.alu_op = ALU.ADD
        elif op == 0x0C:  # ANDI
            c.reg_write = 1; c.alu_src = 2; c.alu_op = ALU.AND
        elif op == 0x0D:  # ORI
            c.reg_write = 1; c.alu_src = 2; c.alu_op = ALU.OR
        elif op == 0x0E:  # XORI
            c.reg_write = 1; c.alu_src = 2; c.alu_op = ALU.XOR
        elif op == 0x0A:  # SLTI
            c.reg_write = 1; c.alu_src = 1; c.alu_op = ALU.SLT
        elif op == 0x0F:  # LUI
            c.reg_write = 1; c.alu_src = 1; c.alu_op = ALU.LUI
        elif op == 0x23:  # LW
            c.reg_write = 1; c.alu_src = 1; c.mem_to_reg = 1; c.alu_op = ALU.ADD
        elif op == 0x2B:  # SW
            c.alu_src = 1; c.mem_write = 1; c.alu_op = ALU.ADD
        elif op == 0x04:  # BEQ
            c.branch = 1; c.alu_op = ALU.SUB
        elif op == 0x02:  # J
            c.jump = 1
        elif op == 0x03:  # JAL
            c.jump = 1; c.reg_write = 1
        return c

# ─────────────────────────────────────────────────────────────────────────────
# Lab 6 — Branch Predictor (1-bit, 2-bit, and BHT)
# ─────────────────────────────────────────────────────────────────────────────

class BranchPredictor:
    """
    Branch History Table with configurable 1-bit or 2-bit saturating
    counters, indexed by PC (Lab 6).
    """

    def __init__(self, bits: int = 2, table_size: int = 64):
        self.bits = bits
        self.table_size = table_size
        # Initialize all counters to weakly not-taken (0)
        self.table: Dict[int, int] = {}
        self.stats_correct = 0
        self.stats_total = 0

    def _index(self, pc: int) -> int:
        return (pc >> 2) % self.table_size

    def predict(self, pc: int) -> bool:
        """Return True if branch is predicted taken."""
        idx = self._index(pc)
        state = self.table.get(idx, 0)
        if self.bits == 1:
            return state == 1
        else:  # 2-bit
            return state >= 2  # Weakly/Strongly taken

    def update(self, pc: int, actually_taken: bool, predicted_taken: bool):
        """Update the predictor after a branch resolves."""
        idx = self._index(pc)
        state = self.table.get(idx, 0)
        max_val = (1 << self.bits) - 1

        if actually_taken:
            state = min(state + 1, max_val)
        else:
            state = max(state - 1, 0)
        self.table[idx] = state

        # Track accuracy
        self.stats_total += 1
        if predicted_taken == actually_taken:
            self.stats_correct += 1

    @property
    def accuracy(self) -> float:
        if self.stats_total == 0:
            return 0.0
        return self.stats_correct / self.stats_total

# ─────────────────────────────────────────────────────────────────────────────
# Lab 7 — 4-Way Set-Associative Cache
# ─────────────────────────────────────────────────────────────────────────────

class CacheLine:
    __slots__ = ("valid", "tag", "data")

    def __init__(self):
        self.valid = False
        self.tag = 0
        self.data = [0] * 4  # 4 words per block (16 bytes)


class Cache:
    """
    4-way set-associative cache (Lab 7).
    32-bit addresses, 16 sets, 16-byte blocks, round-robin replacement.
    """

    NUM_WAYS = 4
    NUM_SETS = 16
    BLOCK_WORDS = 4  # 4 words = 16 bytes per block

    def __init__(self):
        self.sets: List[List[CacheLine]] = [
            [CacheLine() for _ in range(self.NUM_WAYS)]
            for _ in range(self.NUM_SETS)
        ]
        self.repl_ptr = [0] * self.NUM_SETS  # round-robin pointer per set
        self.hits = 0
        self.misses = 0

    def _decompose(self, addr: int) -> Tuple[int, int, int]:
        """Returns (tag, set_index, word_offset)."""
        word_offset = (addr >> 2) & 0x3        # 2 bits for word within block
        set_index   = (addr >> 4) & 0xF        # 4 bits for set
        tag         = (addr >> 8)               # remaining bits
        return tag, set_index, word_offset

    def read(self, addr: int) -> Tuple[bool, int]:
        """Returns (hit, data). On miss, allocates a line and returns 0."""
        tag, idx, woff = self._decompose(addr)
        for line in self.sets[idx]:
            if line.valid and line.tag == tag:
                self.hits += 1
                return True, to_unsigned_32(line.data[woff])

        # Miss — allocate via round-robin
        self.misses += 1
        way = self.repl_ptr[idx]
        self.repl_ptr[idx] = (way + 1) % self.NUM_WAYS
        line = self.sets[idx][way]
        line.valid = True
        line.tag = tag
        line.data = [0] * self.BLOCK_WORDS
        return False, 0

    def write(self, addr: int, value: int) -> bool:
        """Write a word. Returns True on hit, False on miss (allocates)."""
        tag, idx, woff = self._decompose(addr)
        for line in self.sets[idx]:
            if line.valid and line.tag == tag:
                line.data[woff] = to_unsigned_32(value)
                self.hits += 1
                return True

        # Miss — allocate then write
        self.misses += 1
        way = self.repl_ptr[idx]
        self.repl_ptr[idx] = (way + 1) % self.NUM_WAYS
        line = self.sets[idx][way]
        line.valid = True
        line.tag = tag
        line.data = [0] * self.BLOCK_WORDS
        line.data[woff] = to_unsigned_32(value)
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Lab 8 — Two-Level Page Table Walker
# ─────────────────────────────────────────────────────────────────────────────

class PageTableWalker:
    """
    Two-level page table walker (Lab 8).
    Virtual address: [VPN1 (10b)][VPN2 (10b)][Offset (12b)]
    PTE format: [Valid(1)][Dirty(1)][Ref(1)][Write(1)][Read(1)][..PFN..]
    """

    BASE_REGISTER = 0x3FFBFF  # base of L1 page table (22 bits)

    def __init__(self, physical_memory: Dict[int, int]):
        self.memory = physical_memory

    def translate(self, virtual_addr: int, is_write: bool = False
                  ) -> Tuple[Optional[int], str]:
        """
        Walk the page table. Returns (physical_addr, error_string).
        error_string is empty on success.
        """
        vpn1   = (virtual_addr >> 22) & 0x3FF
        vpn2   = (virtual_addr >> 12) & 0x3FF
        offset = virtual_addr & 0xFFF

        # Level 1 lookup
        l1_addr = (self.BASE_REGISTER << 10) | vpn1
        l1_entry = self.memory.get(to_unsigned_32(l1_addr), 0)
        l1_valid = (l1_entry >> 31) & 1
        if not l1_valid:
            return None, "PAGE_FAULT_L1: L1 PTE invalid"

        l1_pfn = l1_entry & 0x3FFFFF  # 22-bit pointer to L2

        # Level 2 lookup
        l2_addr = (l1_pfn << 10) | vpn2
        l2_entry = self.memory.get(to_unsigned_32(l2_addr), 0)
        l2_valid = (l2_entry >> 31) & 1
        if not l2_valid:
            return None, "PAGE_FAULT_L2: L2 PTE invalid"

        l2_write = (l2_entry >> 28) & 1   # write-protect bit
        l2_read  = (l2_entry >> 27) & 1   # read permission

        if not l2_read and not is_write:
            return None, "PROTECTION_FAULT: no read permission"
        if l2_write and is_write:
            return None, "PROTECTION_FAULT: write-protected"

        l2_pfn = l2_entry & 0xFFFFF  # 20-bit physical frame
        physical_addr = (l2_pfn << 12) | offset
        return to_unsigned_32(physical_addr), ""

# ─────────────────────────────────────────────────────────────────────────────
# Lab 9 — Reorder Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ROBEntry:
    __slots__ = ("valid", "pending", "preg", "value")

    def __init__(self):
        self.valid = False
        self.pending = False
        self.preg = 0
        self.value = 0


class ReorderBuffer:
    """
    Circular reorder buffer for in-order commit (Lab 9).
    16 entries, tracks alloc → writeback → commit flow.
    """

    SIZE = 16

    def __init__(self):
        self.entries = [ROBEntry() for _ in range(self.SIZE)]
        self.alloc_ptr = 0
        self.commit_ptr = 0
        self.committed_log: List[Tuple[int, int, int]] = []  # (slot, preg, value)

    @property
    def ready(self) -> bool:
        return not self.entries[self.alloc_ptr].valid

    def allocate(self, preg: int) -> Optional[int]:
        """Allocate a slot. Returns slot index or None if full."""
        if self.entries[self.alloc_ptr].valid:
            return None  # full
        slot = self.alloc_ptr
        e = self.entries[slot]
        e.valid = True
        e.pending = True
        e.preg = preg
        e.value = 0
        self.alloc_ptr = (self.alloc_ptr + 1) % self.SIZE
        return slot

    def writeback(self, slot: int, value: int = 0):
        """Mark a slot as completed (no longer pending)."""
        e = self.entries[slot]
        if e.valid and e.pending:
            e.pending = False
            e.value = value

    def try_commit(self) -> Optional[Tuple[int, int, int]]:
        """
        Commit the head entry if it is valid and not pending.
        Returns (slot, preg, value) or None.
        """
        e = self.entries[self.commit_ptr]
        if e.valid and not e.pending:
            result = (self.commit_ptr, e.preg, e.value)
            e.valid = False
            self.committed_log.append(result)
            self.commit_ptr = (self.commit_ptr + 1) % self.SIZE
            return result
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Lab 5 — Pipeline Registers
# ─────────────────────────────────────────────────────────────────────────────

class PipelineReg:
    """Generic pipeline latch that can be flushed."""

    def __init__(self):
        self.data: dict = {}

    def write(self, **kwargs):
        self.data.update(kwargs)

    def read(self, key: str, default=0):
        return self.data.get(key, default)

    def flush(self):
        self.data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Lab 5 — Forwarding Unit
# ─────────────────────────────────────────────────────────────────────────────

class ForwardingUnit:
    """
    Data-hazard forwarding logic (Lab 5).
    Returns the forwarded value for a source register.
    """

    @staticmethod
    def resolve(reg: int, reg_val: int,
                xm_reg_write: bool, xm_rd: int, xm_alu_result: int,
                mw_reg_write: bool, mw_rd: int, mw_write_data: int) -> int:
        """
        Check EX/MEM and MEM/WB stages for forwarding opportunities.
        Priority: EX/MEM > MEM/WB > register file value.
        """
        if xm_reg_write and xm_rd != 0 and xm_rd == reg:
            return xm_alu_result
        if mw_reg_write and mw_rd != 0 and mw_rd == reg:
            return mw_write_data
        return reg_val

# ─────────────────────────────────────────────────────────────────────────────
# Unified Pipelined MIPS CPU
# ─────────────────────────────────────────────────────────────────────────────

class MIPSCPU:
    """
    5-stage pipelined MIPS CPU with:
      - Instruction decode & control (Labs 2-3)
      - ALU (Labs 1, 2, 3)
      - Data forwarding (Lab 5)
      - Branch prediction (Lab 6)
      - L1 data cache (Lab 7)
      - Page table walker (Lab 8) — available for address translation
      - Reorder buffer (Lab 9) — tracks instruction commit order
    """

    def __init__(self, verbose: bool = False):
        # Memories
        self.i_mem: Dict[int, int] = {}   # instruction memory (word-addressed)
        self.d_mem: Dict[int, int] = {}   # data memory (word-addressed)
        self.rf: List[int] = [0] * 32     # register file ($0 always 0)

        # Pipeline registers
        self.fd = PipelineReg()  # Fetch → Decode
        self.dx = PipelineReg()  # Decode → Execute
        self.xm = PipelineReg()  # Execute → Memory
        self.mw = PipelineReg()  # Memory → Writeback

        # PC
        self.pc = 0

        # Sub-components
        self.branch_predictor = BranchPredictor(bits=2, table_size=64)
        self.cache = Cache()
        self.rob = ReorderBuffer()
        self.clock = Clock()

        # Stats
        self.cycle_count = 0
        self.instr_count = 0
        self.stall_count = 0
        self.flush_count = 0
        self.verbose = verbose
        self.halted = False

    # ── Memory helpers ──────────────────────────────────────────────────

    def load_program(self, instructions: List[int], base_addr: int = 0):
        """Load instruction words into i_mem starting at base_addr (byte)."""
        for i, instr in enumerate(instructions):
            self.i_mem[base_addr + i * 4] = to_unsigned_32(instr)

    def mem_read_word(self, byte_addr: int) -> int:
        """Read a word from data memory through the cache."""
        addr = to_unsigned_32(byte_addr) & ~0x3  # word-align
        hit, cached_val = self.cache.read(addr)
        if hit:
            return cached_val
        # On miss, fetch from backing store and fill cache
        val = self.d_mem.get(addr, 0)
        self.cache.write(addr, val)
        return val

    def mem_write_word(self, byte_addr: int, value: int):
        """Write a word to data memory and cache (write-through)."""
        addr = to_unsigned_32(byte_addr) & ~0x3
        value = to_unsigned_32(value)
        self.d_mem[addr] = value
        self.cache.write(addr, value)

    # ── Hazard detection ────────────────────────────────────────────────

    def _detect_load_use_hazard(self, inst: DecodedInstruction) -> bool:
        """Stall if the previous instruction is a LW and we need its result."""
        if not self.dx.read("ctrl"):
            return False
        dx_ctrl: ControlSignals = self.dx.read("ctrl")
        dx_rd = self.dx.read("write_reg", 0)
        if dx_ctrl.mem_to_reg and dx_rd != 0:
            if dx_rd == inst.rs or (not inst.is_r_type and dx_rd == inst.rt) or \
               (inst.is_r_type and dx_rd == inst.rt):
                return True
        return False

    # ── Pipeline stages ─────────────────────────────────────────────────

    def _stage_fetch(self) -> dict:
        """IF: Fetch instruction at PC from instruction memory."""
        raw = self.i_mem.get(self.pc, 0)
        pc_plus_4 = to_unsigned_32(self.pc + 4)
        return {"raw": raw, "pc": self.pc, "pc_plus_4": pc_plus_4}

    def _stage_decode(self, fetched: dict) -> dict:
        """ID: Decode instruction, read registers, generate control signals."""
        raw = fetched.get("raw", 0)
        inst = DecodedInstruction(raw)
        ctrl = ControlSignals.decode(inst)

        rs_val = self.rf[inst.rs]
        rt_val = self.rf[inst.rt]

        write_reg = inst.rd if ctrl.reg_dst else inst.rt
        if ctrl.jump and inst.op == 0x03:  # JAL writes $ra
            write_reg = 31

        return {
            "inst": inst, "ctrl": ctrl,
            "rs_val": rs_val, "rt_val": rt_val,
            "pc": fetched.get("pc", 0),
            "pc_plus_4": fetched.get("pc_plus_4", 0),
            "write_reg": write_reg,
        }

    def _stage_execute(self, decoded: dict) -> dict:
        """EX: ALU operation, branch resolution, forwarding."""
        inst: DecodedInstruction = decoded.get("inst", DecodedInstruction(0))
        ctrl: ControlSignals = decoded.get("ctrl", ControlSignals())
        pc = decoded.get("pc", 0)
        pc_plus_4 = decoded.get("pc_plus_4", 0)

        # Forwarding
        rs_val = ForwardingUnit.resolve(
            inst.rs, decoded.get("rs_val", 0),
            bool(self.xm.read("ctrl") and self.xm.read("ctrl").reg_write),
            self.xm.read("write_reg", 0), self.xm.read("alu_result", 0),
            bool(self.mw.read("ctrl") and self.mw.read("ctrl").reg_write),
            self.mw.read("write_reg", 0), self.mw.read("write_data", 0),
        )
        rt_val = ForwardingUnit.resolve(
            inst.rt, decoded.get("rt_val", 0),
            bool(self.xm.read("ctrl") and self.xm.read("ctrl").reg_write),
            self.xm.read("write_reg", 0), self.xm.read("alu_result", 0),
            bool(self.mw.read("ctrl") and self.mw.read("ctrl").reg_write),
            self.mw.read("write_reg", 0), self.mw.read("write_data", 0),
        )

        # ALU input B selection
        if ctrl.alu_src == 1:
            alu_b = to_unsigned_32(inst.imm_sign)
        elif ctrl.alu_src == 2:
            alu_b = inst.imm_zero
        else:
            alu_b = rt_val

        alu_result, zero = ALU.execute(rs_val, alu_b, ctrl.alu_op, inst.shamt)

        # Branch target
        branch_target = to_unsigned_32(pc_plus_4 + (inst.imm_sign << 2))
        take_branch = ctrl.branch and zero

        # Jump target
        jump_target = (pc_plus_4 & 0xF0000000) | (inst.addr << 2) if ctrl.jump else 0

        # JAL: result is pc+4 (return address)
        if ctrl.jump and inst.op == 0x03:
            alu_result = pc_plus_4

        return {
            "inst": inst, "ctrl": ctrl,
            "alu_result": alu_result, "rt_val": rt_val,
            "write_reg": decoded.get("write_reg", 0),
            "take_branch": take_branch, "branch_target": branch_target,
            "jump": ctrl.jump, "jump_target": jump_target,
            "pc": pc,
        }

    def _stage_memory(self, executed: dict) -> dict:
        """MEM: Data memory read/write through cache."""
        ctrl: ControlSignals = executed.get("ctrl", ControlSignals())
        alu_result = executed.get("alu_result", 0)
        rt_val = executed.get("rt_val", 0)

        mem_data = 0
        if ctrl.mem_to_reg:  # LW
            mem_data = self.mem_read_word(alu_result)
        if ctrl.mem_write:   # SW
            self.mem_write_word(alu_result, rt_val)

        return {
            "ctrl": ctrl,
            "alu_result": alu_result,
            "mem_data": mem_data,
            "write_reg": executed.get("write_reg", 0),
        }

    def _stage_writeback(self, mem_out: dict):
        """WB: Write result back to register file."""
        ctrl: ControlSignals = mem_out.get("ctrl", ControlSignals())
        if not ctrl.reg_write:
            return

        write_reg = mem_out.get("write_reg", 0)
        if write_reg == 0:
            return  # $0 is hardwired to 0

        if ctrl.mem_to_reg:
            data = mem_out.get("mem_data", 0)
        else:
            data = mem_out.get("alu_result", 0)

        self.rf[write_reg] = to_unsigned_32(data)

        # Store for forwarding reference
        self.mw.write(write_data=to_unsigned_32(data))

    # ── Main cycle ──────────────────────────────────────────────────────

    def step(self):
        """Execute one pipeline cycle."""
        if self.halted:
            return

        self.cycle_count += 1
        self.clock.tick()

        # ── Writeback (from previous cycle's MEM/WB latch) ──
        if self.mw.data:
            self._stage_writeback(self.mw.data)
            # ROB: commit any ready instructions
            self.rob.try_commit()

        # ── Memory (from previous cycle's EX/MEM latch) ──
        mem_out = {}
        if self.xm.data:
            mem_out = self._stage_memory(self.xm.data)

        # ── Execute (from previous cycle's ID/EX latch) ──
        ex_out = {}
        if self.dx.data:
            ex_out = self._stage_execute(self.dx.data)

        # ── Decode (from previous cycle's IF/ID latch) ──
        dec_out = {}
        if self.fd.data:
            dec_out = self._stage_decode(self.fd.data)

        # ── Fetch ──
        fetch_out = self._stage_fetch()

        # ── Hazard handling ──
        stall = False
        flush = False
        new_pc = to_unsigned_32(self.pc + 4)

        # Load-use hazard detection
        if dec_out:
            inst = dec_out.get("inst", DecodedInstruction(0))
            if not inst.is_nop and self._detect_load_use_hazard(inst):
                stall = True
                self.stall_count += 1

        # Branch / Jump resolution
        if ex_out:
            if ex_out.get("take_branch"):
                new_pc = ex_out["branch_target"]
                flush = True
                self.flush_count += 1
                # Update branch predictor
                self.branch_predictor.update(
                    ex_out.get("pc", 0), True,
                    self.branch_predictor.predict(ex_out.get("pc", 0))
                )
            elif ex_out.get("jump"):
                new_pc = ex_out["jump_target"]
                flush = True
                self.flush_count += 1
            else:
                ex_ctrl = ex_out.get("ctrl", ControlSignals())
                if ex_ctrl.branch:
                    self.branch_predictor.update(
                        ex_out.get("pc", 0), False,
                        self.branch_predictor.predict(ex_out.get("pc", 0))
                    )

        # ── Advance pipeline latches ──
        self.mw.data = mem_out if mem_out else {}
        self.xm.data = ex_out if ex_out else {}

        if stall:
            # Insert bubble into EX, keep ID and IF the same
            self.dx.flush()
            # Don't update PC or IF/ID
        elif flush:
            # Squash IF/ID and ID/EX
            self.fd.flush()
            self.dx.flush()
            self.pc = new_pc
        else:
            self.dx.data = dec_out if dec_out else {}
            self.fd.data = fetch_out
            self.pc = new_pc

        # Track instruction count (non-NOP instructions entering decode)
        if dec_out and not stall:
            inst = dec_out.get("inst", DecodedInstruction(0))
            if not inst.is_nop:
                self.instr_count += 1
                # ROB: allocate slot for every instruction that writes a register
                ctrl = dec_out.get("ctrl", ControlSignals())
                if ctrl.reg_write:
                    slot = self.rob.allocate(dec_out.get("write_reg", 0))
                    if slot is not None:
                        # Immediately writeback for this simple model
                        self.rob.writeback(slot)

        # Halt detection: if instruction memory is empty at PC
        if self.pc not in self.i_mem and not stall:
            # Drain pipeline for a few more cycles
            pass

        if self.verbose:
            self._print_state()

    def run(self, max_cycles: int = 1000):
        """Run until the pipeline drains or max_cycles is reached."""
        # We consider the CPU halted when the PC has gone past all
        # loaded instructions and the pipeline is drained.
        drain = 0
        for _ in range(max_cycles):
            self.step()
            if self.pc not in self.i_mem:
                drain += 1
                if drain >= 5:  # let pipeline flush
                    break
            else:
                drain = 0

    # ── Debug / display ─────────────────────────────────────────────────

    def _print_state(self):
        print(f"  [Cycle {self.cycle_count:4d}]  PC={self.pc:#010x}  "
              f"Clock={self.clock}")

    def dump_registers(self):
        print("\n═══ Register File ═══")
        for i in range(0, 32, 4):
            regs = "  ".join(
                f"${i+j:<2d}={self.rf[i+j]:#010x}" for j in range(4)
            )
            print(f"  {regs}")

    def dump_memory(self, limit: int = 32):
        print("\n═══ Data Memory (non-zero) ═══")
        count = 0
        for addr in sorted(self.d_mem.keys()):
            if self.d_mem[addr] != 0:
                print(f"  [{addr:#010x}] = {self.d_mem[addr]:#010x}  "
                      f"({to_signed_32(self.d_mem[addr])})")
                count += 1
                if count >= limit:
                    print("  ... (truncated)")
                    break
        if count == 0:
            print("  (empty)")

    def dump_stats(self):
        print("\n═══ Simulation Statistics ═══")
        print(f"  Total cycles:         {self.cycle_count}")
        print(f"  Instructions:         {self.instr_count}")
        if self.instr_count > 0:
            print(f"  CPI:                  {self.cycle_count / self.instr_count:.2f}")
        print(f"  Pipeline stalls:      {self.stall_count}")
        print(f"  Pipeline flushes:     {self.flush_count}")
        print(f"  Branch predictor acc:  {self.branch_predictor.accuracy:.1%}")
        total_cache = self.cache.hits + self.cache.misses
        if total_cache > 0:
            print(f"  Cache hits / total:   {self.cache.hits}/{total_cache} "
                  f"({self.cache.hits/total_cache:.1%})")
        print(f"  ROB commits:          {len(self.rob.committed_log)}")
        print(f"  Sim clock:            {self.clock}")

# ─────────────────────────────────────────────────────────────────────────────
# Demo program
# ─────────────────────────────────────────────────────────────────────────────

def demo_program() -> List[int]:
    """
    A small MIPS program that exercises most supported instructions:

        addi $t0, $zero, 5       # $t0 = 5
        addi $t1, $zero, 10      # $t1 = 10
        add  $t2, $t0, $t1       # $t2 = 15
        sub  $t3, $t1, $t0       # $t3 = 5
        and  $t4, $t2, $t3       # $t4 = 5
        or   $t5, $t0, $t1       # $t5 = 15
        slt  $t6, $t0, $t1       # $t6 = 1  (5 < 10)
        sw   $t2, 0($zero)       # mem[0] = 15
        lw   $t7, 0($zero)       # $t7 = 15
        addi $t0, $t0, -1        # $t0 = 4
        beq  $t0, $zero, +2      # not taken (4 != 0)
        addi $t0, $t0, -4        # $t0 = 0
        lui  $t8, 0x1234         # $t8 = 0x12340000
        ori  $t8, $t8, 0x5678    # $t8 = 0x12345678
    """
    return [
        0x20080005,  # addi $t0, $zero, 5
        0x2009000A,  # addi $t1, $zero, 10
        0x01095020,  # add  $t2, $t0, $t1
        0x01285822,  # sub  $t3, $t1, $t0
        0x014B6024,  # and  $t4, $t2, $t3
        0x01096825,  # or   $t5, $t0, $t1
        0x0109702A,  # slt  $t6, $t0, $t1
        0xAC0A0000,  # sw   $t2, 0($zero)
        0x8C0F0000,  # lw   $t7, 0($zero)
        0x2108FFFF,  # addi $t0, $t0, -1
        0x11000002,  # beq  $t0, $zero, +2
        0x2108FFFC,  # addi $t0, $t0, -4
        0x3C181234,  # lui  $t8, 0x1234
        0x37185678,  # ori  $t8, $t8, 0x5678
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MIPS CPU Simulator"
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to a hex file with one instruction per line")
    parser.add_argument("--cycles", "-n", type=int, default=200,
                        help="Maximum simulation cycles (default 200)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print state every cycle")
    args = parser.parse_args()

    cpu = MIPSCPU(verbose=args.verbose)

    if args.file:
        instructions = []
        with open(args.file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    instructions.append(int(line, 16))
        cpu.load_program(instructions)
        print(f"Loaded {len(instructions)} instructions from {args.file}")
    else:
        prog = demo_program()
        cpu.load_program(prog)
        print(f"Running built-in demo program ({len(prog)} instructions)\n")

    cpu.run(max_cycles=args.cycles)

    cpu.dump_registers()
    cpu.dump_memory()
    cpu.dump_stats()

    # Quick sanity checks for the demo program
    if not args.file:
        print("\n═══ Demo Assertions ═══")
        checks = [
            (cpu.rf[10], 15,          "$t2 = 15 (5 + 10)"),
            (cpu.rf[11], 5,           "$t3 = 5  (10 - 5)"),
            (cpu.rf[12], 5,           "$t4 = 5  (15 & 5)"),
            (cpu.rf[13], 15,          "$t5 = 15 (5 | 10)"),
            (cpu.rf[14], 1,           "$t6 = 1  (5 < 10)"),
            (cpu.rf[15], 15,          "$t7 = 15 (loaded from mem[0])"),
            (cpu.rf[24], 0x12345678,  "$t8 = 0x12345678 (LUI+ORI)"),
            (cpu.d_mem.get(0, -1), 15,"mem[0] = 15"),
        ]
        all_pass = True
        for actual, expected, desc in checks:
            status = "✓" if to_unsigned_32(actual) == to_unsigned_32(expected) else "✗"
            if status == "✗":
                all_pass = False
            print(f"  {status}  {desc}  "
                  f"(got {to_unsigned_32(actual):#010x}, "
                  f"expected {to_unsigned_32(expected):#010x})")

        if all_pass:
            print("\n  All checks passed")
        else:
            print("\n  Some checks failed — debug with --verbose")


if __name__ == "__main__":
    main()
