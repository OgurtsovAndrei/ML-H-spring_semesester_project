from functools import wraps
import time
import struct


def with_time(func):
    # @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class Timer:
    def __enter__(self):
        # Code to be executed when entering the block
        print("Running with ")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Code to be executed when exiting the block
        end_time = time.perf_counter()
        total_time = end_time - self.start_time
        print(f'Took {total_time:.4f} seconds')
        pass

    def execute(self, code):
        # Execute the provided code within the block
        exec(code)


class FloatBitsConverter:
    @staticmethod
    def get_float_bits_in_memory(value: float, split_to_mantissa_and_exp=False):
        # Pack the float value as a binary string
        binary = struct.pack('d', value)

        # Convert the binary string to a list of bits
        bits = []
        for byte in binary:
            bits.extend([bool(byte & (1 << i)) for i in range(7, -1, -1)])

        # Return the list of bits
        if split_to_mantissa_and_exp:
            return bits[0], bits[1:12], bits[1:12]

        return bits

    @staticmethod
    def get_float_by_bits(bits: list[bool]):
        assert (len(bits) == 64)
        # Convert the list of bits to a binary string
        # Pad the binary string with leading zeros to ensure it has a multiple of 8 bits
        binary = ''.join('1' if bit else '0' for bit in bits)
        binary = binary.zfill((len(binary) + 7) // 8 * 8)
        # Convert the binary string back to a byte string
        byte_string = bytes(int(binary[i:i + 8], 2) for i in range(0, len(binary), 8))
        # Unpack the byte string as a float value
        value: float = struct.unpack('d', byte_string)[0]

        return value


# Usage:
if __name__ == '__main__':
    bits = FloatBitsConverter.get_float_bits_in_memory(3.1415)
    print(bits)
    print(FloatBitsConverter.get_float_by_bits(bits))