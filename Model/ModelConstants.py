sentence_len = 512

dataset_params = {'batch_size': 256,
          'shuffle': True}

tokenizer_config = {
    'max_length': sentence_len,
    'pad_to_max_length': True,
    'truncation_strategy': 'only_second',
    'return_tensors': 'pt'
}
