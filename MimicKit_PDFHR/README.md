# MimicKit_PDFHR

`MimicKit_PDFHR` contains the PDF-HR experiments built on the
[MimicKit](README_MimicKit.md) repository.

## Scope

The release provides matched baseline and PDF-HR experiments for four motions:


| Motion    | ADD config                           | ADD+PDF-HR config                      |
| --------- | ------------------------------------ | -------------------------------------- |
| backflip  | `args/args/add_g1_backflip_args.txt` | `args/add_g1_backflip_PDFHR_args.txt`  |
| sideflip  | `args/add_g1_sideflip_args.txt`      | `args/add_g1_sideflip_PDFHR_args.txt`  |
| spinkick  | `args/add_g1_spinkick_args.txt`      | `args/add_g1_spinkick_PDFHR_args.txt`  |
| cartwheel | `args/add_g1_cartwheel_args.txt`     | `args/add_g1_cartwheel_PDFHR_args.txt` |


## Assets and motion files

Follow the [MimicKit README](README_MimicKit.md) to download the G1 and G1 motion assets and place them in the specified directories.

## Training

Train the ADD baseline:

```bash
./run.sh 0 args/add_g1_backflip_args.txt
```

Train the matched ADD+PDF-HR experiment:

```bash
./run.sh 0 args/add_g1_backflip_PDFHR_args.txt
```

## Testing

Test a trained policy with viser viewer:

```bash
./test.sh 0 output/add_g1_backflip_PDFHR
```

Test a specified checkpoint:

```bash
./test.sh 0 output/add_g1_backflip_PDFHR --model <checkpoint>
```

Use `--model <checkpoint>` to select `model.pt`, an intermediate checkpoint ID, or an explicit checkpoint path.

## Upstream

This project is built on [MimicKit](README_MimicKit.md). See
[UPSTREAM.md](UPSTREAM.md) for the imported revision.
