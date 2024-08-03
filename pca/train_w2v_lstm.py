def main(args):
    from . import train_w2v
    train_w2v.main(args)

    from . import train_lstm
    return train_lstm.main(args)