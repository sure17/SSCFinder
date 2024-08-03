import sys


def main():
    try:
        from . import options
        opts = options.Options(sys.argv[1:])
        assert opts, 'main.py-main-failed to parse cmdline args'

        args = opts.args()
        assert args, 'main.py-main-failed to get cmdline args'
        assert args.cmd, 'main.py-main-failed to get null action'

        if args.cmd == "train-w2v":
            from . import train_w2v
            return train_w2v.main(args)
        
        elif args.cmd == "train-lstm":
            from . import train_lstm
            return train_lstm.main(args)

        elif args.cmd == "train-w2v-lstm":
            from . import train_w2v_lstm
            return train_w2v_lstm.main(args)
        
        elif args.cmd == "predict":
            from . import inference
            return inference.main(args)


    except Exception as e:
        print("Exception Occured!!!")
        print(str(e))
        exit(1)


if __name__ == '__main__':
    main()