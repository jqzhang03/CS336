import os
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = None

# 创建基于GPT2的正则表达式的分词模式
def get_gpt2_regex():
    global GPT2_REGEX
    if GPT2_REGEX is None:
        import regex as re

        GPT2_REGEX = re.compile(GPT2_PRETOKENIZE_PATTERN)

# 计算text中出现的word和其出现的频率
def count_word_freq_from_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    if not text:
        return {}
    spans = [text]
    for special_token in special_tokens: 
        new_spans: list[str] = []
        for sp in spans:
            if sp:
                new_spans.extend(sp.split(special_token))
        spans = new_spans
    word_freq: dict[tuple[bytes, ...], int] = {}
    for sp in spans:
        if not sp:
            continue
        for m in GPT2_REGEX.finditer(sp):
            piece = m.group(0)
            if not piece:
                continue
            byts = piece.encode("utf-8")
            key = tuple(bytes([b]) for b in byts)
            word_freq[key] = word_freq.get(key, 0) + 1
    return word_freq

# 子进程根据分配的路径、起始位置、special_token进行划分，然后返回文本中出现的word和频率
def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8", errors = "ignore")
    return count_word_freq_from_text(text, special_tokens)

# 单进程任务不需要分块，因此跟无优化版本操作一样
def build_word_freq_serial(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    get_gpt2_regex()
    with open(input_path, "r", encoding = "utf-8") as f:
        text = f.read()
    return count_word_freq_from_text(text, special_tokens)

# 多进程任务需要分块
def build_word_freq_parallel(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int) -> dict[tuple[bytes, ...], int]:
    # 如果没有special_token的话其实也是跟单进程任务一样执行
    # 因为我们是根据special_token进行划分
    if num_processes <= 1 or not special_tokens:
        return build_word_freq_serial(input_path, special_tokens)
    split_special_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        # find_chunk_boundaries在pretokenization_example.py中已经给出，因此不用实现
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
    
    # 定义分块后的任务，开始的位置，结束的位置，spcial_token
    tasks = [(str(input_path), start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    merged = Counter()
    # 为每一个子进程都定义一份自己的分词模式，避免rx在进程间反复通信
    with Pool(processes = num_processes, initializer = get_gpt2_regex) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize = 1):
            merged.update(partial)

    return dict(merged)

# 计算一个word中相邻字符出现的频率
def pairs_in_word(word: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    if len(word) < 2:
        return counts
    prev = word[0]
    for cur in word[1:]:
        p = (prev, cur)
        counts[p] = counts.get(p, 0) + 1
        prev = cur
    return counts

def apply_merge(word: tuple[bytes, ...], a: bytes, b: bytes, new_token: bytes) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word
    merged: list[bytes] = []
    i = 0
    L = len(word)
    while i < L:
        if i < L - 1 and word[i] == a and word[i + 1] == b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def build_pair_stats(word_freq: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word, freq in word_freq.items():
        if len(word) < 2:
            continue
        local = pairs_in_word(word)
        for p, occ in local.items():
            pair_counts[p] = pair_counts.get(p, 0) + occ * freq
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)
    return pair_counts, pair_to_words

def remove_word_contribution(word: tuple[bytes, ...], freq: int, pair_counts: dict[tuple[bytes, bytes], int], pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]) -> None:
    local = pairs_in_word(word)
    for pair, occ in local.items():
        # 得到当前这个字节对都出现在哪些word里
        s = pair_to_words.get(pair)
        if s is not None:
            s.discard(word)
            if not s:
                del pair_to_words[pair]
        new_c = pair_counts.get(pair, 0) - occ * freq
        if new_c <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_c

def add_word_contribution(word: tuple[bytes, ...], add_freq: int, pair_counts: dict[tuple[bytes, bytes], int], pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], *, word_is_new: bool) -> None:
    if len(word) < 2:
        return
    local = pairs_in_word(word)
    for p, occ in local.items():
        pair_counts[p] = pair_counts.get(p, 0) + occ * add_freq
        if word_is_new:
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_precesses: int | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size <= 0 or vocab_size < 256 + len(special_tokens): 
        raise ValueError("vocab_size is wrong.")
    
    # vocab是一个int映射bytes的字典，首先把单字节全部映射到他的int类型
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    # 然后需要把定义的special_tokens添加到字典的后面，然后再添加新token
    for token in special_tokens: 
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # 如果没有定义进程数，那么就获取cpu的数量，如果cpu数量不确定则使用一个进程
    # cpu数量意味着我们并行运算可以创建多少个子进程
    if num_precesses is None:
        num_precesses = min(8, os.cpu_count() or 1)

    # 获取文件大小，如果文件小到一定程度，分块没有意义，甚至会对运行速度起负作用
    file_size = os.path.getsize(input_path)
    if num_precesses <= 1 or file_size < 1000000:
        word_freq = build_word_freq_serial(input_path, special_tokens)
    else:
        word_freq = build_word_freq_parallel(input_path, special_tokens, num_precesses)
    
    if not word_freq: 
        return vocab, []

    # 进入循环之前先初始化，记录字节对出现的次数以及字节对出现在哪些word当中
    pair_counts, pair_to_words = build_pair_stats(word_freq)
    merges: list[tuple[bytes, bytes]] = []
    
    while next_id < vocab_size:
        if not pair_counts:
            break
        
        # 找到pair_counts中出现频率最高的一对，排序的规则就是如果频率一样，字典序大的在前面
        # kv[1]也就是说比较最大的，先比较频率，然后是kv[0]，比较字典序大的
        (a, b), best_count = max(pair_counts.items(), key = lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break

        # 把a和b拼接起来放到字典里
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        # 对于a和b这一字节对来说，查询其影响了哪些word，我们只需要修改受影响的word即可，不需要把整个语料库进行遍历
        affected = pair_to_words.get((a, b))
        if not affected:
            pair_counts.pop((a, b), None)
            continue
        
        add_back: dict[tuple[bytes, ...], int] = {}
        for word in list(affected):
            freq = word_freq.get(word)
            if freq is None:
                continue

            remove_word_contribution(word, freq, pair_counts, pair_to_words)
            del word_freq[word]

            new_word = apply_merge(word, a, b, new_token)
            add_back[new_word] = add_back.get(new_word, 0) + freq
        for new_word, add_freq in add_back.items():
            existed = new_word in word_freq
            word_freq[new_word] = word_freq.get(new_word, 0) + add_freq
            add_word_contribution(new_word, add_freq, pair_counts, pair_to_words, word_is_new = not existed)

    return vocab, merges