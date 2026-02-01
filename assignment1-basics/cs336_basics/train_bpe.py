import os

GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = None

def get_gpt2_regex():
    global GPT2_REGEX
    if GPT2_REGEX is None:
        import regex as re

        GPT2_REGEX = re.compile(GPT2_PRETOKENIZE_PATTERN)
    return GPT2_REGEX

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
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
    
    # 原始BPE只是根据空格来划分每个word，这里运用GPT-2所使用的基于正则表达式的预分词器
    # 它的作用是把一大段切成一个个的分块，word_freq来记录语料库中的每个token以及其出现的次数
    rx = get_gpt2_regex()
    word_freq: dict[tuple[bytes, ...], int] = {}
    # 从input_path中读取文件内容
    with open(input_path, "r", encoding = "utf-8") as f: 
        text = f.read()
    
    if not text: 
        return vocab, []

    # 定义一个spans列表，首先把我们得到的text放进去，然后根据输入的special_tokens把text进行分块
    # 得到去掉special_tokens后的每个小片段
    spans = [text]
    for special_token in special_tokens: 
        new_spans: list[str] = []
        for sp in spans: 
            if sp:
             -   new_spans.extend(sp.split(special_token))
        spans = new_spans

    # 对于去掉special_tokens的每个小片段，将其进行预分词处理
    for sp in spans:
        if not sp:
            continue
        
        # 基于rx的匹配规则，将sp进行预分词
        for match in rx.finditer(sp):
            # piece就是当前匹配到的word
            piece = match.group(0)
            if not piece:
                continue

            # 将匹配到的word转变为字节串，变成b'....'
            byts = piece.encode("utf-8")
            # 将字节串按字节一个一个划分成单字节token序列
            key = tuple(bytes([b]) for b in byts)
            # dict的函数get(key, default)，如果能找到key就返回value值，否则返回default，在这里也就是0
            word_freq[key] = word_freq.get(key, 0) + 1
        
    # 记录每一步的合并规则
    merges: list[tuple[bytes, bytes]] = []
    
    while next_id < vocab_size:
        # 遍历每一个token序列，得到相邻的两个token出现的次数
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for word, freq in word_freq.items():
            # len<2意味着只有一个字符，没有相邻的
            if len(word) < 2:
                continue
            prev = word[0]
            for cur in word[1:]:
                pair = (prev, cur)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                prev = cur
        # pair_counts为空，说明无法继续merge了
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

        # 更新新的token序列和频率
        new_word_freq: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freq.items():
            if len(word) < 2:
                new_word_freq[word] = new_word_freq.get(word, 0) + freq
                continue

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
            
            key = tuple(merged)
            new_word_freq[key] = new_word_freq.get(key, 0) + freq

        word_freq = new_word_freq

    return vocab, merges