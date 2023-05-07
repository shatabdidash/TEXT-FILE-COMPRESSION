import heapq
import collections


def calculate_probabilities(file_path):
    # Open the file in read mode
    with open(file_path, "r") as file:
        # Count the frequency of each character in the file
        frequency = collections.Counter(file.read())
        # Calculate the probability of each character
        total_count = sum(frequency.values())
        probabilities = {symbol: count / total_count for symbol, count in frequency.items()}
    return probabilities


def shannon_fano(probabilities, prefix=""):
    # Sort the symbols based on their probabilities
    symbols = sorted(probabilities, key=probabilities.get, reverse=True)
    # Base case: if there's only one symbol, assign the prefix 0
    if len(symbols) == 1:
        codebook = {symbols[0]: prefix + "0"}
        return codebook
    # Divide the symbols into two groups with approximately equal probabilities
    mid = len(symbols) // 2
    left_symbols = symbols[:mid]
    right_symbols = symbols[mid:]
    # Recursively generate the codebook for the left and right groups
    left_codebook = shannon_fano({symbol: probabilities[symbol] for symbol in left_symbols}, prefix + "0")
    right_codebook = shannon_fano({symbol: probabilities[symbol] for symbol in right_symbols}, prefix + "1")
    # Merge the codebooks for the left and right groups
    codebook = {**left_codebook, **right_codebook}
    return codebook


def encode_file(file_path, codebook):
    # Open the file in read mode and create an output file in write mode
    with open(file_path, "r") as input_file, open(file_path + ".sf", "w") as output_file:
        # Encode each symbol in the input file using the codebook and write the encoded data to the output file
        encoded_data = "".join(codebook[symbol] for symbol in input_file.read())
        output_file.write(encoded_data)


def decode_file(file_path, codebook):
    # Open the encoded file in read mode and create an output file in write mode
    with open(file_path, "r") as input_file, open(file_path[:-3] + ".decoded", "w") as output_file:
        # Invert the codebook so that the codes are keys and the symbols are values
        inverse_codebook = {code: symbol for symbol, code in codebook.items()}
        # Decode the encoded data and write the decoded symbols to the output file
        decoded_data = ""
        code = ""
        while True:
            # Read one bit at a time from the input file
            bit = input_file.read(1)
            if not bit:
                # End of file
                break
            # Add the bit to the code and check if it matches a symbol in the codebook
            code += bit
            if code in inverse_codebook:
                decoded_data += inverse_codebook[code]
                code = ""
        output_file.write(decoded_data)


def main():
    file_path = input("Enter the path to the text file: ")
    probabilities = calculate_probabilities(file_path)
    codebook = shannon_fano(probabilities)
    encode_file(file_path, codebook)
    decode_file(file_path + ".sf", codebook)


if __name__ == "__main__":
    main()
