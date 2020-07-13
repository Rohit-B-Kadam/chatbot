import sys


def print_help():
    print("Provide following command line argument: ")
    print("--train : for training")
    print("--eval  : for evaluate")


def main():
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "--train":
            import models.joint_bert.train_joint_bert
        elif sys.argv[1].lower() == "--eval":
            import models.joint_bert.eval_joint_bert
        else:
            print_help()
    else:
        print_help()


if __name__ == "__main__":
    main()
