# Documents downloader

Since this bot was created internally at the Ethereum foundation, the `download_ethereum.sh` bash script has been made available to make it easier to instantiate DocuBot targeted at Ethereum specific documents.

This directory can serve as a placeholder for other use-cases. If you intend to add a download script, please submit a PR accordingly.

## Ethereum downloader

To run the ethereum script, provide the desired output directory name as an argument

```bash
./download_ethereum.sh <output_directory_name>
```

Replace `<output_directory_name>` with the desired name for the output directory. For example

```bash
./scripts/bash/download_ethereum.sh ethereum-docs
```

You can then point DocuBot to `ethereum-docs` to instantiate an Ethereum specific chatbot.
