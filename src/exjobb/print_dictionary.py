import pprint
import pickle

FILE = 'garage_1.dictionary'

def main():
    with open(FILE, 'rb') as cached_pcd_file:
        cache_data = pickle.load(cached_pcd_file)
        pprint.pprint(cache_data)
    return

if __name__ == "__main__":
    main()
