#!/usr/bin/env python3
"""
Example: projecting a minimal Internal Hypothesis Graph event into a VCF v4.3 record.
This is a toy example for schema clarity (not a full caller).
"""

from typing import Dict, Any, List

def vcf_header() -> str:
    lines = []
    lines.append("##fileformat=VCFv4.3")
    lines.append('##source=GenomeLTM-MoE')
    lines.append('##INFO=<ID=IHG_ID,Number=1,Type=String,Description="Internal Hypothesis Graph node ID",Source="GenomeLTM",Version="v0">')
    lines.append('##INFO=<ID=POST,Number=A,Type=Float,Description="Posterior probability of ALT allele(s)",Source="GenomeLTM",Version="v0">')
    lines.append('##INFO=<ID=UNC,Number=1,Type=Float,Description="Uncertainty score",Source="GenomeLTM",Version="v0">')
    lines.append('##INFO=<ID=TECHSUP,Number=.,Type=String,Description="Per-technology support summary",Source="GenomeLTM",Version="v0">')
    lines.append('##FILTER=<ID=LOWCONF,Description="Below confidence threshold">')
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    lines.append('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">')
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE")
    return "\n".join(lines)

def project_event(event: Dict[str, Any]) -> str:
    chrom = event["locus"]["chrom"]
    pos = event["locus"]["pos"]
    ihg_id = event["ihg_id"]
    ref = event["alleles"]["ref"]
    alt_list: List[str] = event["alleles"]["alt"]
    alt = ",".join(alt_list) if alt_list else "."
    post = event.get("posterior", 0.0)
    unc = event.get("uncertainty", 0.0)

    qual = int(round(99 * post))  # toy mapping; real QUAL should be calibrated
    filt = "PASS" if post >= 0.99 else "LOWCONF"

    techsup_parts = []
    for tech, sup in (event.get("per_tech_support") or {}).items():
        techsup_parts.append(f"{tech}:DP={sup.get('dp',0)};CONF={sup.get('conf',0.0):.3f}")
    techsup = ",".join(techsup_parts) if techsup_parts else "."

    info = f"IHG_ID={ihg_id};POST={post:.6f};UNC={unc:.6f};TECHSUP={techsup}"
    fmt = "GT:GQ"
    sample = "0/1:99"  # placeholder

    return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\t{fmt}\t{sample}"

if __name__ == "__main__":
    example_event = {
        "ihg_id": "IHG_0000001",
        "event_type": "SNV",
        "locus": {"chrom": "1", "pos": 1234567},
        "alleles": {"ref": "A", "alt": ["G"]},
        "posterior": 0.995,
        "uncertainty": 0.02,
        "per_tech_support": {
            "ILLUMINA": {"dp": 38, "conf": 0.992},
            "PACBIO": {"dp": 12, "conf": 0.997},
            "ONT": {"dp": 18, "conf": 0.981}
        }
    }
    print(vcf_header())
    print(project_event(example_event))
