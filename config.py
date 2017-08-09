dict(
    # Sequences longer than this will be deleted
    seq_len_src=10,
    seq_len_trg=10,
    # Batch size of train data
    batch_size=80,
    # vocabulary size, include '</S>'
    src_vocab_size=30002,
    trg_vocab_size=30002,

    # Vocabularies
    # vocab_src='./data/vocab_src.pkl',
    # vocab_trg='./data/vocab_trg.pkl',
    # Datasets
    train_src='./data/train_src.shuffle',
    train_trg='./data/train_trg.shuffle',
    valid_src='./data/valid_src',
    valid_trg='./data/valid_trg',
    valid_out='./data/valid_out',

    sample_freq=400,
    
    replace_unk=False,

    # Maxmum number of epoch
    finish_after=1,
    # Save model after this many updates
    save_freq=5000,
    # Valid frequence
    valid_freq=400,
    valid_freq_fine=5000,
    # Start bleu validation after this many updates
    val_burn_in=1,
    val_burn_in_fine=150000,
)
