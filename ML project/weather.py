# ml_algorithms.py

# Sample training data
dataset = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# -------------------------------
# FIND-S ALGORITHM
# -------------------------------
def find_s_algorithm(data):
    num_attributes = len(data[0]) - 1
    hypothesis = ['0'] * num_attributes

    for example in data:
        if example[-1] == 'Yes':
            for i in range(num_attributes):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    return hypothesis


# -------------------------------
# CANDIDATE ELIMINATION ALGORITHM
# -------------------------------
def is_more_general(hypothesis, instance):
    return all(h == '?' or h == val for h, val in zip(hypothesis, instance))


def update_S(S, instance):
    new_S = S[:]
    for i in range(len(S)):
        if new_S[i] == '0':
            new_S[i] = instance[i]
        elif new_S[i] != instance[i]:
            new_S[i] = '?'
    return new_S


def update_G(G, S, instance):
    new_G = []
    for g in G:
        if is_more_general(g, instance):
            for i in range(len(g)):
                if g[i] == '?':
                    if S[i] != instance[i]:
                        new_hypothesis = g[:]
                        new_hypothesis[i] = S[i]
                        if new_hypothesis not in new_G:
                            new_G.append(new_hypothesis)
    return new_G


def candidate_elimination_algorithm(data):
    num_attributes = len(data[0]) - 1
    S = ['0'] * num_attributes
    G = [['?' for _ in range(num_attributes)]]

    for example in data:
        inputs = example[:-1]
        output = example[-1]

        if output == 'Yes':
            G = [g for g in G if is_more_general(g, inputs)]
            S = update_S(S, inputs)
        else:
            G = update_G(G, S, inputs)

    return S, G


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("Training Data:")
    for row in dataset:
        print(row)

    # Run Find-S
    print("\n=== Find-S Algorithm ===")
    final_hypothesis_find_s = find_s_algorithm(dataset)
    print("Final Hypothesis (Find-S):", final_hypothesis_find_s)

    # Run Candidate Elimination
    print("\n=== Candidate Elimination Algorithm ===")
    final_S, final_G = candidate_elimination_algorithm(dataset)
    print("Final Specific Hypothesis (S):", final_S)
    print("Final General Hypotheses (G):", final_G)
