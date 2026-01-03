"""
bamRenameTags.py
Marcus Viscardi,    May 23, 2024

I really want to be able to take a BAM file with custom auxiliary tags and rename them.

This is important for my work here because I need to be able to run the umi_tools group function first to collapse
B30s, then I need to use it again to collapse the UMIs.

Unfortunately, the second time I run it, it will overwrite the B30s that I have already collapsed. So I need to rename
the first round of tags to something else before running the second round!

I think this should be easy just by using pysam to read in the BAM file and then
write it back out with the new tag names.
"""
import pysam
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from tqdm.auto import tqdm
import shutil

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument("bam_file", type=Path,
                       help="Path to the BAM file to rename tags in")
# Can I make an option for users to give a list of tags to rename?
argparser.add_argument("old_tag_names", type=str,
                       help="Old tag names to replace, separated by commas")
argparser.add_argument("new_tag_names", type=str,
                       help="New tag names to use, separated by commas")
argparser.add_argument("--out_file", type=Path,
                       help="Path to write the new BAM file to. If not provided, we will overwrite the input file.",
                       default=None)

def rename_bam_tags(bam_file: Path, old_tag_names, new_tag_names, out_file=None):
    """
    
    :param bam_file: Path to the SAM/BAM file to rename tags in
    :param old_tag_names: Comma seperated list of old tag names to replace (as a string)
    :param new_tag_names: Comma seperated list of new tag names to use (as a string)
    :param out_file: Path to write the new SAM/BAM file to. If not provided, we will overwrite the input file.
    :return: Path of the new SAM/BAM file
    """
    old_tag_names = old_tag_names.split(",")
    new_tag_names = new_tag_names.split(",")
    assert bam_file.exists(), f"Could not find file: {bam_file}"
    assert bam_file.suffix in [".bam", ".cram", ".sam"], "Input file must be a BAM, SAM, or CRAM file"
    assert new_tag_names is not None, "You must provide new tag names"
    assert old_tag_names is not None, "You must provide old tag names"
    assert len(new_tag_names) == len(old_tag_names), ("Number of new tag names must match number of old tag names\n"
                                                      f"\tOld tag names: {old_tag_names}\n"
                                                      f"\tNew tag names: {new_tag_names}\n")
    print(f"Renaming tags: {old_tag_names} to {new_tag_names}")
    
    overwrite = False
    if out_file is None:
        overwrite = True
        tmp_file = NamedTemporaryFile(suffix=bam_file.suffix, delete=False)
        print(f"Writing to temporary file: {tmp_file.name}")
        print(f"Will overwrite {bam_file}")
        out_file = Path(tmp_file.name)

    write_mode = "wb" if bam_file.suffix == ".bam" else "w"
    with pysam.AlignmentFile(bam_file) as bam_in:
        total_reads = bam_in.count()
    
    with (pysam.AlignmentFile(bam_file) as bam_in,
          pysam.AlignmentFile(out_file, write_mode, template=bam_in) as bam_out):
        for read in tqdm(bam_in.fetch(), total=total_reads, desc=f"Renaming tags in {bam_file.name}"):
            read_tag_names = [tag_name for tag_name, tag_value in read.tags]
            for new_tag, old_tag in zip(new_tag_names, old_tag_names):
                if old_tag in read_tag_names:
                    read.tags = [(new_tag, value) if tag == old_tag else (tag, value) for tag, value in read.tags]
            bam_out.write(read)
    
    if overwrite:
        shutil.move(out_file, bam_file)
        print(f"Renamed tags in {bam_file}")
    else:
        print(f"Renamed tags in {bam_file} and saved to {bam_file}")
    return out_file

if __name__ == '__main__':
    
    args = argparser.parse_args()
    rename_bam_tags(args.bam_file, args.old_tag_names, args.new_tag_names, args.out_file)
    
