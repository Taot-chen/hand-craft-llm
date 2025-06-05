from tokenizer import Tokenizer

def main():
    enc = Tokenizer('./tok4096.model')
    text = 'Hello, world!'
    print(enc.encode(text, bos=True, eos=True))
    print(enc.decode(enc.encode(text, bos=True, eos=True)))

if __name__ == "__main__":
    main()

# OUTPUT:
# [1, 346, 2233, 4010, 1475, 4021, 2]
# Hello, world!