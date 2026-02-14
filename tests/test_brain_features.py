import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.brain import SmartBrain

def test_brain_features():
    print("🤖 Initializing SDU AI Agent (SmartBrain)...")
    try:
        brain = SmartBrain()
    except Exception as e:
        print(f"❌ Failed to initialize Brain: {e}")
        return

    query = "จุดเด่นของ ม.สวนดุสิต คืออะไร"
    print(f"\n🔍 Testing Query: '{query}'")

    # 1. Test Retrieval
    print("\n📚 [1] Testing Retrieval (Vector Search)...")
    candidates = brain.retrieve(query, top_k=5)
    print(f"   -> Retrieved {len(candidates)} candidates.")
    for i, c in enumerate(candidates[:2]):
        print(f"      - Candidate {i+1}: {c['text'][:100]}... (Score: {c['score']:.4f})")

    # 2. Test Reranking
    print("\n⭐ [2] Testing Reranking (Gemini)...")
    reranked = brain.rerank(query, candidates, top_n=3)
    print(f"   -> Reranked to {len(reranked)} top results.")
    for i, c in enumerate(reranked):
        meta = c['metadata']
        source = f"{meta.get('source', 'Unknown')} (Page {meta.get('page','-')})"
        print(f"      - Rank {i+1}: {c['text'][:100]}... [Source: {source}]")

    # 3. Test Memory (History Management)
    print("\n🧠 [3] Testing Memory Management...")
    history = [
        {"role": "user", "content": "สวัสดีครับ"},
        {"role": "assistant", "content": "สวัสดีครับ มีอะไรให้พี่ช่วยไหมครับ"},
        {"role": "user", "content": "ขอถามเรื่องค่าเทอมหน่อย"},
        {"role": "assistant", "content": "ค่าเทอมขึ้นอยู่กับคณะครับ"},
    ]
    # We can't easily see internal state, but we can dry-run the think method
    context = "\n".join([c['text'] for c in reranked])
    print("   -> Sending query with history...")
    response = brain.think(query, context, history)
    print(f"   -> Response: {response['text']}")
    if response['usage']:
        print(f"   -> Usage: {response['usage']}")

if __name__ == "__main__":
    test_brain_features()
