import sys
import pickle

if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as handle:
        data = pickle.load(handle)
        print(data)