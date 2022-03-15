This is a Pytorch based implementation of Locally Asynchronous Parallel SGD and Locally Partitioned Asynchronous Parallel SGD algorithms as described in paper [1] -- please cite the paper if you use or derive from this code.

To run the code run the scripts from samplescripts folder.

The three sample scripts can be used in three different 
settings: single node, multi-node, and slurm, as their names suggest.

Set the variables, the directory paths, etc in the runscripts according to
the available systems settings.

This code was developed on Python 3.5 and has been tested on Python 3.7 based on Anaconda installation. Mainly, the requirement includes the latest Pytorch and Numpy packages. 
The requirements.txt file can be used to install the required dependencies.

[1] Scaling the Wild: Decentralizing Hogwild!-style Shared-memory SGD, Bapi Chatterjee and Vyacheslav Kungurtsev and Dan Alistarh, arXiv, 2022.
Cite it as 
@misc{chatterjee2022scaling,
      title={Scaling the Wild: Decentralizing Hogwild!-style Shared-memory SGD}, 
      author={Bapi Chatterjee and Vyacheslav Kungurtsev and Dan Alistarh},
      year={2022},
      eprint={2203.06638},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
