# BAMnet


Code accompanying the paper ["Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases"](https://arxiv.org/abs/1903.02188)


# Get started

## Train the BAMnet model
`python train.py -config [bamnet_config]`

## Test the BAMnet model
`python test.py -config [bamnet_config]`

## Train the topic entity predictor
`python train_entnet.py -config [entnet_config]`

## Test the topic entity predictor
`python test_entnet.py -config [entnet_config]`

## Test the whole system (BAMnet + topic entity predictor)
`python test_pipeline.py -bamnet_config [bamnet_config] -entnet_config [entnet_config] -raw_data [raw_data_dir]`


# Reference

If you found this code useful, please cite the following paper:

Yu Chen, Lingfei Wu, Mohammed J. Zaki. **"Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases."** *In Proc. 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT2019). June 2019.*

@article{chen2019bidirectional,
  title={Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases},
  author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J},
  journal={arXiv preprint arXiv:1903.02188},
  year={2019}
}
