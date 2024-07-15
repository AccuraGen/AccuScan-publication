import argparse
import polars as pl
import gc
import pyBigWig
import logging
import math
from pathlib import Path
from io import StringIO


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_depth_df(species_depth_p, bed_file_p, nrows=None):
    """Read in depth from BigWig object and store in dataframe"""
    bw = pyBigWig.open(species_depth_p)
    depth_values = []
    with open(bed_file_p, "r") as input_file:
        for row_num, line in enumerate(input_file):
            cols = line.rstrip().split("\t")
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            # Grab the depth from each position within the interval
            depths = [0 if math.isnan(x) else int(x) for x in bw.values(chrom, start, end + 1)]
            for i, depth in enumerate(depths):
                depth_values.append([chrom, i + start, depth])

            # Optionally break early for testing purposes
            if nrows and row_num > nrows:
                break
    logger.info("Finished reading depth from bigwig file")
    depth_headers = ["chrom", "pos", "dp"]
    depth_df = pl.DataFrame(depth_values, schema=depth_headers).lazy()
    # Delete depth_values after it has been used to clear up memory
    logger.info("Finished created dataframe")
    del depth_values
    gc.collect()
    logger.info("Finished reading depth")
    return depth_df


def get_vcf_df(confirmed_variant_vcf_p):
    """Read in vcf as polars dataframe, which is used to annotate uc of genotype variants."""
    vcf_columns = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "subject"]
    vcf_df = pl.read_csv(
        confirmed_variant_vcf_p, separator="\t", comment_prefix="#", has_header=False, new_columns=vcf_columns
    )
    # Pull out UC from info field
    vcf_df = vcf_df.with_columns(pl.col("info").str.extract(";UC=(.*?);").cast(int).alias("uc"))
    vcf_df = vcf_df[["chrom", "pos", "ref", "alt", "uc"]]

    # Convert to lazyframe to enable joining with the genotype_df
    vcf_df = vcf_df.lazy()
    logger.info("Finished reading vcf")
    return vcf_df


def add_error_rate(df):
    """Calculate error_rate by dividing the dp by uc. If uc is 0, then set error_rate to INF."""
    return df.with_columns((pl.col("dp") / pl.col("uc")).fill_nan("INF").alias("error_rate"))


def process_genotypes(genotypes_df, depth_df, vcf_df, min_dp, max_uc):
    """Parse genotype variants with incorporating dp/uc to calculate initial error rate"""

    # Join with depth_df to grab the dp
    genotypes_df = genotypes_df.join(depth_df, on=["chrom", "pos"], how="left")

    # Join with vcf_df to grab the uc
    genotypes_df = genotypes_df.join(vcf_df, on=["chrom", "pos", "ref", "alt"], how="left")
    genotypes_df = genotypes_df.with_columns(pl.col("uc").fill_null(0))

    # Filter for only the rows that meet necessary criteria
    genotypes_df = genotypes_df.filter(
        (pl.col("dp") > min_dp)
        & (pl.col("uc") <= max_uc)
    )

    # Summarize fp rate
    stats = (
        genotypes_df.group_by("Type")
        .agg(pl.col("dp").sum(), pl.col("uc").sum())
         .fill_null(0)
         .sort("Type")
    )
    # Delete genotypes_df after it has been used to clear up memory
    del genotypes_df
    gc.collect()

    stats = add_error_rate(stats)
    stats = stats.collect()
    return stats


def final_transformation(stats, output_file):
    """Convert from initial error rate to finalized error rate"""

    final = pl.DataFrame(stats)

    # Create additional row of non C-T variant overall false positive rate
    overall_error = final.sum()
    overall_error[0, "Type"] = "overall"
    final = pl.concat([final, overall_error])

    final = add_error_rate(final)


    final.write_csv(output_file, separator="\t")
    logger.info(f"All done. Final output written to {output_file}")
    return final


def main(vcf, depth_file, genotype_v_f, bed_file, output_file, min_dp, max_uc, nrows=None):
    """Perform all transformations needed to calculate the error rate for any given firefly directory"""

    logger.info(f"Start processing ")

    logger.info(f"Reading in depth information from {depth_file} and {bed_file}")
    depth_df = get_depth_df(depth_file, bed_file, nrows=nrows)

    logger.info(f"Reading in vcf information from {vcf}")
    vcf_df = get_vcf_df(vcf)
    
    genotypes_df = pl.scan_csv(genotype_v_f, separator="\t",n_rows=nrows)
    stats = process_genotypes(genotypes_df, depth_df, vcf_df, min_dp, max_uc)

    final_transformation(stats, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate error rate for given sample")
    parser.add_argument("--vcf",  required=True, help="Name of the panel for error rate calculation")
    parser.add_argument("--depth_file", required=True, help="Name of the panel for error rate calculation")
    parser.add_argument("--output_file", required=True, help="Name of the panel for error rate calculation")
    parser.add_argument("--genotypes_tsv", required=True, help="Path to tsv containing variants used for error rate calculation")
    parser.add_argument("--bed_file", required=True, help="Path to bedfile for error rate panel regions")
    parser.add_argument("--nrows", type=int, help="Optionally parse fewer rows for testing.")
    parser.add_argument("--min_dp", type=int, default=15, help="mininum depth to be considered in plasma.")
    parser.add_argument("--max_uc", type=int, default=1, help="maximum molecule count (unique count = uc) to be considered.")
    args = parser.parse_args()

    main(args.vcf, args.depth_file, args.genotypes_tsv, args.bed_file, args.output_file, args.min_dp, args.max_uc, nrows=args.nrows)
