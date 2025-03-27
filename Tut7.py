import numpy as np
from itertools import combinations
from collections import Counter

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Initialize Apriori algorithm parameters.

        Parameters:
        - min_support: Minimum support threshold (float between 0 and 1)
        - min_confidence: Minimum confidence threshold (float between 0 and 1)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}  # Store frequent itemsets by length
        self.association_rules = []  # Store association rules

    def _get_support(self, itemset, transactions):
        """Calculate the support of an itemset."""
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / len(transactions)

    def _generate_candidates(self, prev_itemsets, length):
        """Generate candidate itemsets of a specific length."""
        items = sorted(set(item for itemset in prev_itemsets for item in itemset))
        return [frozenset(comb) for comb in combinations(items, length)]

    def _prune_candidates(self, candidates, prev_itemsets, length):
        """Prune candidates using the Apriori property (all subsets must be frequent)."""
        return [
            candidate for candidate in candidates
            if all(frozenset(subset) in prev_itemsets for subset in combinations(candidate, length - 1))
        ]

    def fit(self, transactions):
        """
        Mine frequent itemsets and generate association rules.

        Parameters:
        - transactions: List of transactions (each transaction is a list of items)
        """
        # Convert transactions to frozensets for efficient lookup
        transactions = [frozenset(t) for t in transactions]
        item_counts = Counter(item for transaction in transactions for item in transaction)

        # Get frequent 1-itemsets
        candidates = {frozenset([item]) for item, count in item_counts.items()
                      if count / len(transactions) >= self.min_support}

        length = 1
        while candidates:
            self.frequent_itemsets[length] = [(itemset, self._get_support(itemset, transactions))
                                              for itemset in candidates]

            # Generate new candidates of length (length + 1)
            length += 1
            candidates = self._generate_candidates(candidates, length)
            candidates = self._prune_candidates(candidates, {i[0] for i in self.frequent_itemsets[length - 1]}, length - 1)

            # Filter by min support
            candidates = {itemset for itemset in candidates
                          if self._get_support(itemset, transactions) >= self.min_support}

        # Generate association rules
        self.association_rules = []
        for length, itemsets in self.frequent_itemsets.items():
            if length == 1:
                continue

            for itemset, support in itemsets:
                for i in range(1, length):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        if not consequent:
                            continue

                        confidence = support / self._get_support(antecedent, transactions)
                        if confidence >= self.min_confidence:
                            self.association_rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence
                            })

    def get_frequent_itemsets(self, length=None):
        """Get frequent itemsets."""
        if length is None:
            return [itemset for lst in self.frequent_itemsets.values() for itemset in lst]
        return self.frequent_itemsets.get(length, [])

    def get_association_rules(self):
        """Get association rules meeting min confidence threshold."""
        return self.association_rules

# Example usage
if __name__ == "__main__":
    # Sample transaction data (market basket data)
    transactions = [
        ['bread', 'milk'],
        ['bread', 'diapers', 'beer', 'eggs'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['bread', 'milk', 'diapers', 'beer'],
        ['bread', 'milk', 'diapers', 'cola']
    ]

    # Initialize and run Apriori algorithm
    apriori = Apriori(min_support=0.4, min_confidence=0.6)
    apriori.fit(transactions)

    # Display results
    print("\nFrequent Itemsets:")
    for length, itemsets in apriori.frequent_itemsets.items():
        print(f"\nLength {length}:")
        for itemset, support in itemsets:
            print(f"{set(itemset)}: support = {support:.2f}")

    print("\nAssociation Rules:")
    for rule in apriori.get_association_rules():
        print(f"{set(rule['antecedent'])} => {set(rule['consequent'])} "
              f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f})")
