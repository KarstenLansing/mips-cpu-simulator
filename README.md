# MIPS CPU Simulator

A complete 32-bit MIPS processor simulation in pure Python — no external dependencies.

Built as a consolidation of UCSB CS 154 (Computer Architecture) coursework into a single, runnable simulator that models the full lifecycle of instruction execution.

## What's Inside

| Component | Description |
|-----------|-------------|
| **5-Stage Pipeline** | IF → ID → EX → MEM → WB with proper latch propagation |
| **32-bit ALU** | ADD, SUB, AND, OR, XOR, NOR, SLT, SLL, SRL, SRA, LUI |
| **Instruction Decoder** | Full R-type, I-type, and J-type MIPS instruction parsing |
| **Control Unit** | Generates reg_dst, branch, reg_write, alu_src, mem_write, mem_to_reg signals |
| **Data Forwarding** | EX/MEM and MEM/WB forwarding to resolve RAW hazards without stalling |
| **Hazard Detection** | Load-use stalls and branch flush logic |
| **Branch Predictor** | Configurable 1-bit or 2-bit saturating counter BHT with accuracy tracking |
| **4-Way Set-Associative Cache** | 16 sets, 16-byte blocks, round-robin replacement, hit/miss stats |
| **Page Table Walker** | Two-level translation with valid/dirty/ref/read/write permission checks |
| **Reorder Buffer** | 16-entry circular ROB for in-order commit tracking |

## Supported Instructions

`ADD` `SUB` `AND` `OR` `XOR` `NOR` `SLT` `SLL` `SRL` `SRA` `ADDI` `ANDI` `ORI` `XORI` `SLTI` `LUI` `LW` `SW` `BEQ` `J` `JAL`

## Usage

```bash
# Run the built-in demo program
python3 mips_cpu_sim.py

# Load a hex instruction file
python3 mips_cpu_sim.py --file programs/demo.hex

# Verbose mode — prints per-cycle pipeline state
python3 mips_cpu_sim.py --verbose

# Set max simulation cycles
python3 mips_cpu_sim.py --cycles 500
```

## Sample Output

```
Running built-in demo program (14 instructions)

═══ Register File ═══
  $0 =0x00000000  $1 =0x00000000  $2 =0x00000000  $3 =0x00000000
  $8 =0x00000000  $9 =0x0000000a  $10=0x0000000f  $11=0x00000005
  $12=0x00000005  $13=0x0000000f  $14=0x00000001  $15=0x0000000f
  $24=0x12345678  ...

═══ Simulation Statistics ═══
  Total cycles:         18
  Instructions:         14
  CPI:                  1.29
  Pipeline stalls:      0
  Pipeline flushes:     0
  Branch predictor acc: 100.0%
  Cache hits / total:   1/2 (50.0%)
  ROB commits:          12
```

## Hex Instruction Format

Create a text file with one 32-bit hex instruction per line:

```
20080005
2009000A
01095020
01285822
```

Lines starting with `#` are treated as comments.

## Project Structure

```
mips-cpu-simulator/
├── mips_cpu_sim.py       # The simulator
├── programs/
│   └── demo.hex          # Sample program (hex instructions)
└── README.md
```

## Requirements

- Python 3.6+
- No external dependencies

## Author

**Karsten Lansing** — UC Santa Barbara, B.S. Computer Science
