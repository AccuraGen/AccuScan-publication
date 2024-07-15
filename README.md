# README

The repository includes two scripts that are supplementing the publication of AccuScan: “Ultra-sensitive molecular residual disease detection through whole genome sequencing with single-read error correction”.  
Input files for the scripts below can be found on request at https://zenodo.org/records/11188534

The scripts are:

### mrd_caller.py
script to call MRD status given a file with information about
* how many somatic variants were detected
* how many molecules were investigated (sum of depth at all somatic variant positions)
* error rate per variant type
The output 
#### Usage
`python mrd_caller.py <sample_name> <path_to_mrd_input_file> `

additional options:
```
positional arguments:
  sample_id             Sample ID
  infile                Filename with counts for depth, error rate and observations

options:
  -h, --help            show this help message and exit
  --output_file OUTPUT_FILE
                        Output filename, if different from a transformation of the input filename
  --error_format ERROR_FORMAT
                        Format for expressing error rate, per_base is the average number of instances per error
  --specificity SPECIFICITY
                        Specificity (1-type1 error rate)
  --confidence CONFIDENCE
                        Confidence value for CI and upper bound
```

### errorrate_per_sample.py
script to calculate the error rate per variant_type given variant calls from a plasma sample and molecule depth information at each position. Only a subset of ~200M positions is evaluated. 
#### Requirements
* We ran the script on a machine with 256GB of memory. 
* polars-0.20.26 (can typically be installed with `pip install polars`)
* pybigwig (can typically be installed with `pip install pybigwig`)
#### Usage
`python errorrate_per_sample.py  --vcf <vcf_file> --depth_file <depth_bigwig_file> --genotypes_tsv <uncompressed_default_variant_file> --bed_file <default_region_bed_file> --output_file <output_file>`
#### example
`python errorrate_per_sample.py  --vcf healthy1.vcf.gz --depth_file healthy1.depth.bw --genotypes_tsv default_errorrate_variants.tsv --bed_file default_errorrate_region.bed --output_file /tmp/test_error.txt`

