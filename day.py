import re
from collections import Counter
from typing import List

def get_top_3_words(paragraph: str) -> List [str]:
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'her', 'his', 'i', 'in', 'is', 'it', 'its', 'me',
        'my', 'not', 'of', 'on', 'or', 'she', 'that', 'the', 'their',
        'they', 'this', 'to', 'was', 'we', 'with', 'you', 'our', 'your',
        'but', 'so', 'if', 'then', 'than', 'do', 'does', 'did', 'could',
        'would', 'should', 'may', 'might', 'must', 'very', 'just', 'like',
        'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'
    }
    text_lower = paragraph.lower()
    words = re.findall(r'\b[a-z\'-]+\b', text_lower)
    filtered_words = [w for w in words if w not in stop_words and len(w) > 0 and not all(c in "\'-" for c in w)]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(3)
    if not top_words:
        return []
    result = []
    for word, count in top_words:
        result.append((word,count))
    result.sort(key=lambda x: (-x[1], x[0]))
    return [word for word, _ in result[:3]]

def main():
    print("=== Top 3 Most Frequent Words (Excluding Stop WOrds) ===\n")

    test_paragraphs = [
        """The quick brown fox jumps over the lazy dog. The fox is quick and 
        the dog is lazy. But the fox jumps high while the dog sleeps.""",
        
        """To be or not to be, that is the question. Whether 'tis nobler in 
        the mind to suffer the slings and arrows of outrageous fortune.""",
        
        """Python is amazing. Python is powerful. Python is versatile. 
        Python programming is fun and exciting.""" 
    ]

    for i, paragraph in enumerate(test_paragraphs, 1):
        print(f"Paragraph {i}:")
        print(f'"{paragraph[:100]}{"..." if len(paragraph) > 100 else ""}"')
        print(f"Top 3 words: {get_top_3_words(paragraph)}")
        print("-" * 50)


if __name__ == "__main__":
    # Interactive mode
    print("Enter your paragraph (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "" and not lines:
            continue
        if line == "":
            break
        lines.append(line)
    
    if lines:
        paragraph = "\n".join(lines)
        result = get_top_3_words(paragraph)
        
        if result:
            print(f"\nTop 3 most frequent words (excluding stop words): {result}")
        else:
            print("\nNo words found after filtering stop words.")
    else:
        print("\nNo input provided. Running demo mode...")
        main()