# Values in the CONSEQUENCE column that indicate a variant is unlikely to be of clinical interest.
low_impact_or_non_coding_variant_types = [
    "intron_variant",  # mutation in intron (non-coding), usually benign
    "downstream_gene_variant",  # Variants located downstream of a gene, outside coding or regulatory regions.
    "upstream_gene_variant",  # Variants located upstream of a gene, outside coding or regulatory regions.
    "synonymous_variant",  # Variant codes for the same amino sequence of acids
    "3_prime_UTR_variant",  # Mutation in the 3' UTR (untranslated region) of a gene, which is the region of the gene's mRNA transcript located downstream of the coding sequence, after the stop codon
    "5_prime_UTR_variant",  # A 5' UTR (untranslated region) variant refers to a mutation in the 5' UTR of a gene, which is located upstream of the coding sequence, before the start codon
]
