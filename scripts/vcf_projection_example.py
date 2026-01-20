"""Example script for projecting a VCF record into a smaller schema."""

from typing import Dict


VCF_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]


def project_vcf_record(record: Dict[str, str]) -> Dict[str, str]:
    return {field: record.get(field, "") for field in VCF_FIELDS}


def main() -> None:
    sample = {
        "CHROM": "1",
        "POS": "879317",
        "ID": "rs123",
        "REF": "A",
        "ALT": "G",
        "QUAL": "50",
        "FILTER": "PASS",
        "INFO": "DP=100"
    }
    print(project_vcf_record(sample))


if __name__ == "__main__":
    main()
