from itertools import product
from tqdm import tqdm

# storing string as const to avoid typos
PizzaSlice = "PizzaSlice"
WasabiRoot = "WasabiRoot"
Snowball = "Snowball"
Shells = "Shells"

exchange_rate_dict = {
    PizzaSlice: {PizzaSlice: 1, WasabiRoot: 0.48, Snowball: 1.52, Shells: 0.71},
    WasabiRoot: {PizzaSlice: 2.05, WasabiRoot: 1, Snowball: 3.26, Shells: 1.56}, 
    Snowball: {PizzaSlice: 0.64, WasabiRoot: 0.3 , Snowball: 1, Shells: 0.46}, 
    Shells: {PizzaSlice: 1.41, WasabiRoot: 0.61, Snowball: 2.08, Shells: 1}
}

def iterate_exchanges(total_rounds=5, fx_rate=exchange_rate_dict):
    """
    start with Shells, exchange to A1, then A1->A2->A3->A4, and finally A4->Shells
    there are five rounds of exchanges
    """
    max_val = 0
    max_seq = ""
    nominal = 2_000_000

    for seq in tqdm((product([PizzaSlice, WasabiRoot, Snowball, Shells]
                                         , repeat=(total_rounds - 1)))):
        this_exchanged_val = exchange_rate_dict[Shells][seq[0]] * \
                   exchange_rate_dict[seq[0]][seq[1]] * \
                   exchange_rate_dict[seq[1]][seq[2]] * \
                   exchange_rate_dict[seq[2]][seq[3]] * \
                   exchange_rate_dict[seq[3]][Shells]

        this_exchanged_val *= nominal
    
        if this_exchanged_val > max_val:
            max_val = this_exchanged_val
            max_seq = seq


    return max_val, max_seq



if __name__=="__main__":
    print("performing calculation:")
    max_shells, max_sequence = iterate_exchanges(total_rounds=5)
    print(f"max_shells: {max_shells}\n")
    print(f"max_sequence: Shells, {max_sequence}, Shells")