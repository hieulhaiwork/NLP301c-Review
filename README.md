# NLP301c-Review

### Các thuật ngữ
- affixes: tiền tố hậu tố
- stem: từ gốc
- lexicon: is a vocabulary, a list of words, a dictionary.
- WordNet là lexicon database cho tiếng Anh
- NLG: Natural Language Generation

### Các kĩ thuật tiền xử lí

- Stopword Removing: Loại bỏ stopwords
- Lowercase: Chuyển đổi chữ hoa thành chữ thường
- Tokenization: Tách câu thành các tokens 
- Lemmatization: Chuyển từ về dạng gốc (stem) của nó (ví dụ: "running" → "run"), từ sau khi chuyển đổi là 1 từ có nghĩa
- Stemming: Cắt bớt tiền tố, hậu tố của từ để thu được gốc từ (ví dụ: "running" → "run"), từ sau khi chuyển đổi có thể không có ý nghĩa (happiess → happi)

### Stemming error
- Overstemming (Lỗi rút gọn quá mức)
- Understemming (Lỗi rút gọn chưa đủ)

### Porter stemmer algorithm
Porter Stemmer Algorithm là một thuật toán phổ biến được sử dụng để rút gọn từ (stemming) trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP). Nó giúp chuyển một từ về dạng gốc hoặc dạng căn bản của nó bằng cách loại bỏ các hậu tố (suffixes) nhưng không cố gắng ánh xạ chính xác về từ nguyên gốc.

Đầu ra: Một "stem" (gốc từ) mà không nhất thiết phải là từ đúng về mặt ngữ pháp trong ngôn ngữ tự nhiên.

### N-Gram Stemmer
là một phương pháp rút gọn từ (stemming) dựa trên các N-grams, tức là các chuỗi con có độ dài cố định N, để phân tích và xác định gốc từ. Phương pháp này không dựa trên quy tắc loại bỏ hậu tố như các thuật toán Porter Stemmer hay Snowball Stemmer, mà thay vào đó sử dụng các n-grams để tìm ra các mẫu (patterns) chung giữa các từ có liên quan.

N-Gram: Là một chuỗi con gồm N ký tự liền kề từ một từ. Ví dụ: Với N=3, từ "running" sẽ được chia thành các n-grams: run, unn, nni, nin, ing.

Keywork: language independent

### Native Bayes và SVM

- Naive Bayes là mô hình xác suất dựa trên giả định rằng các từ trong tweet (hoặc các features) độc lập với nhau, điều này có thể không phản ánh đầy đủ sự phức tạp của ngữ cảnh trong các tweet.

- SVM là một mô hình mạnh mẽ có khả năng tạo ra các biên phân cách tối ưu (hyperplane) giữa các lớp dữ liệu. Khi dữ liệu văn bản (như tweet) được chuyển thành vector đặc trưng thông qua các kỹ thuật như TF-IDF hoặc word embeddings (ví dụ như Word2Vec, GloVe, BERT), SVM có thể phân biệt các tweet theo đặc trưng của các lớp dữ liệu và nhóm chúng thành các lớp (ví dụ: tích cực, tiêu cực, trung lập, hoặc các chủ đề khác nhau).

Chốt lại trong 2 mô hình trên không cái nào có khả năng hiểu ngữ cảnh (context) để phân loại text.

### Topic Modeling
Topic Modeling (Mô hình hóa chủ đề) là một kỹ thuật trong xử lý ngôn ngữ tự nhiên (NLP) và học máy (machine learning) dùng để phát hiện các chủ đề ẩn (latent topics) trong một tập hợp các văn bản.

- Latent Dirichlet Allocation (LDA): LDA giả định rằng mỗi tài liệu là sự kết hợp của một số chủ đề (topic), và mỗi chủ đề lại có sự phân phối xác suất của các từ. LDA sẽ tìm ra các chủ đề này dựa trên các từ xuất hiện trong tài liệu.
- Non-Negative Matrix Factorization (NMF): NMF là một kỹ thuật phân tích ma trận, giúp tìm ra các chủ đề bằng cách phân tích các ma trận từ điển từ các tài liệu văn bản.
- Latent Semantic Analysis (LSA): LSA sử dụng phép phân tích thành phần chính (PCA) để tìm ra các mối quan hệ ẩn giữa các từ và các tài liệu, từ đó xác định các chủ đề ẩn trong các tập văn bản.

### LDA:
- Term (Thuật ngữ): Term là một từ trong bộ từ vựng mà mô hình LDA sử dụng để phân tích các tài liệu văn bản. Một "term" có thể là bất kỳ từ nào xuất hiện trong tập hợp dữ liệu văn bản, chẳng hạn như danh từ, động từ, tính từ, hoặc các từ thông dụng. Trong bối cảnh LDA, "term" đơn giản là từ mà mô hình sẽ sử dụng để xác định mối quan hệ giữa các chủ đề và các tài liệu.
- Topic trong LDA là một phân phối xác suất của các từ trong bộ từ vựng. Mỗi topic có thể coi là một tập hợp các từ liên quan với nhau, mà khi xuất hiện trong một tài liệu, chúng có xu hướng chỉ ra rằng tài liệu đó thuộc về chủ đề đó. LDA giả định rằng mỗi tài liệu là sự kết hợp của một số chủ đề, và mỗi chủ đề lại có một phân phối xác suất nhất định cho các từ (terms).
- Document trong LDA là một đơn vị văn bản cần được phân tích. Mỗi tài liệu có thể là một bài báo, một bài viết, một tweet, một email, hoặc bất kỳ đoạn văn bản nào trong tập dữ liệu.

VD:
- Giả sử bạn có 3 tài liệu sau:  
Document 1: "Bệnh nhân mắc ung thư cần điều trị kịp thời."  
Document 2: "Chính phủ tăng cường các biện pháp bảo vệ môi trường."  
Document 3: "Các công ty đầu tư vào công nghệ trí tuệ nhân tạo."  
- LDA có thể tìm ra các chủ đề như:  
Topic 1 (Y tế): Các từ như "bệnh", "chữa trị", "ung thư", "bệnh nhân". (đây là các terms)
Topic 2 (Chính trị): Các từ như "chính phủ", "biện pháp", "bảo vệ", "môi trường".  
Topic 3 (Công nghệ): Các từ như "công ty", "đầu tư", "trí tuệ nhân tạo", "công nghệ".  
- Mỗi tài liệu sẽ được mô hình phân phối với các tỷ lệ khác nhau cho từng chủ đề, ví dụ:  
Document 1: 80% "Y tế", 20% "Chính trị"  
Document 2: 60% "Chính trị", 40% "Y tế"  
Document 3: 90% "Công nghệ", 10% "Chính trị"

Notes: Alpha: density of topics generated within documents, beta: density of terms generated within topics

### TF-IDF
- TF đo lường số lần một từ xuất hiện trong một tài liệu cụ thể so với tổng số từ trong tài liệu đó.
- IDF đánh giá tầm quan trọng của một từ trong toàn bộ tập tài liệu. Nếu một từ xuất hiện trong nhiều tài liệu, thì từ đó không có nhiều giá trị trong việc phân biệt các tài liệu. Giả sử trong một tập dữ liệu D có 100 tài liệu. Từ "data" xuất hiện trong 10 tài liệu. Khi đó, IDF của từ "data" là: IDF("data",D)=log(100/10)

Cuối cùng, TF-IDF là sự kết hợp giữa TF và IDF để xác định mức độ quan trọng của một từ trong một tài liệu so với toàn bộ tập tài liệu. Công thức tính TF-IDF của một từ t
trong tài liệu d:

```TF-IDF(t, d, D) = TF(t, d) x IDF(t, D)```

### Word2Vec
Word2Vec là một mô hình học máy được phát triển để học các biểu diễn từ (word embeddings) có ý nghĩa trong không gian vector.

Có hai phương pháp chính trong việc huấn luyện mô hình Word2Vec:
- Continuous Bag of Words (CBOW): Dự đoán từ trung tâm (target word) dựa trên các từ xung quanh (context words).
- Skip-gram: Dự đoán các từ xung quanh (context words) dựa trên từ trung tâm (target word).

Cả hai mô hình này đều dựa trên một mạng nơ-ron đơn giản với một lớp đầu vào (input layer), một lớp ẩn (hidden layer), và một lớp đầu ra (output layer).

Đặc điểm của Word2Vec:
- Word2Vec học các word embeddings – các vector số học (thường có kích thước từ 100 đến 300 chiều) đại diện cho các từ trong không gian đa chiều.
- Các từ có ý nghĩa tương tự sẽ có các vector gần nhau trong không gian này.
- Tính tương tự (Similarity): Các từ có nghĩa gần giống nhau trong ngữ cảnh sẽ có các vector tương tự. Ví dụ, các từ "king" và "queen" sẽ có các vector gần nhau.
- Tính toán các mối quan hệ (Relationships): Mô hình Word2Vec có thể học được các mối quan hệ toán học giữa các từ. Ví dụ, "king" - "man" + "woman" ≈ "queen". Điều này cho thấy rằng Word2Vec có khả năng nhận diện các mối quan hệ giữa các từ trong không gian vector.
- Để giảm chi phí tính toán, Word2Vec sử dụng các kỹ thuật như negative sampling hoặc hierarchical softmax để tăng tốc quá trình huấn luyện và giảm thời gian tính toán.
  + Negative Sampling: Đây là một kỹ thuật tối ưu hóa nhằm giảm chi phí tính toán của hàm softmax trong mô hình Word2Vec. Thay vì tính toán xác suất cho tất cả các từ trong từ điển, negative sampling chỉ tính toán một số ít từ ngẫu nhiên không có trong ngữ cảnh.
  + Hierarchical Softmax: Là một kỹ thuật thay thế hàm softmax để tính toán xác suất đầu ra, giúp giảm chi phí tính toán cho các mô hình có từ điển lớn.

### Markov Model

Markov Model (MM) là một mô hình toán học dùng để mô phỏng quá trình có tính chất Markov, tức là một quá trình mà trạng thái hiện tại chỉ phụ thuộc vào trạng thái ngay trước đó, không phụ thuộc vào các trạng thái trước đó xa hơn.

Các tasks thông dụng:
- Part-of-Speech (POS) Tagging: Hidden Markov Models (HMM) là một dạng đặc biệt của Markov Model rất phổ biến trong bài toán này. Trong đó, các trạng thái ẩn (hidden states) tương ứng với các nhãn từ loại, còn các quan sát (observations) là các từ trong câu.
- Speech Recognition: mô hình Markov giúp xác định chuỗi các âm vị (phonemes) dựa trên các đặc trưng của tín hiệu đầu vào, với giả định rằng âm vị hiện tại phụ thuộc vào âm vị ngay trước đó.
- Named Entity Recognition (NER)
- Machine Translation (MT): Hidden Markov Models để mô hình hóa sự chuyển đổi giữa các từ hoặc cụm từ trong hai ngôn ngữ. Một trạng thái trong mô hình có thể tương ứng với một từ hoặc cụm từ trong ngôn ngữ nguồn, trong khi các trạng thái kế tiếp đại diện cho các từ hoặc cụm từ trong ngôn ngữ đích.
- Text Generation.

VD: Suppose we want to calculate a probability for the sequence of observations {'Dry','Rain'}. If the following are the possible hidden state sequences, then P('Dry' 'Rain') = ---------.

{'Low', 'Low'}, {'Low', 'High'}, {'High', 'Low'}, and {'High', 'High'}

### HMM - observation likelihoods (Hidden Markov Model)
Observation likelihoods (xác suất quan sát) đo lường khả năng xuất hiện của một quan sát (observation) dựa trên trạng thái ẩn (hidden state)

HMM là một mô hình xác suất được sử dụng để mô tả một chuỗi các quan sát mà các trạng thái ẩn của nó không thể trực tiếp quan sát được, nhưng có thể dựa vào các quan sát để suy luận. HMM bao gồm:

- Các trạng thái ẩn (hidden states): Các trạng thái không thể quan sát trực tiếp.
- Các quan sát (observations): Những dữ liệu mà ta có thể quan sát, thường được gắn với các trạng thái ẩn.
- Chuyển trạng thái: Xác suất chuyển từ trạng thái ẩn này sang trạng thái ẩn khác.
- Xác suất quan sát (observation likelihood): Xác suất một quan sát cụ thể xảy ra trong một trạng thái ẩn nhất định.

Notes:
- The likelihood of a POS tag given the preceding tag: Đây là transition probability (xác suất chuyển trạng thái) chứ không phải xác suất quan sát. Xác suất chuyển trạng thái đo lường khả năng chuyển từ một POS tag này sang POS tag khác.
- The likelihood of a word given a POS tag: Đây là định nghĩa chính xác của observation likelihood trong HMM. Trong HMM, ta quan tâm đến việc tính xác suất của một từ (word) xảy ra khi biết trạng thái ẩn (POS tag) là gì. Ví dụ, khi biết POS tag là "noun" (danh từ), ta tính xác suất xuất hiện của một từ như "dog".

### Thuật toán Viterbi 
là một thuật toán động được sử dụng để tìm chuỗi trạng thái tối ưu trong các mô hình xác suất chuỗi, đặc biệt là trong Hidden Markov Models (HMM). Mục tiêu của thuật toán Viterbi là xác định chuỗi trạng thái ẩn có xác suất cao nhất (tức là chuỗi trạng thái tối ưu) cho một chuỗi quan sát (observations) đã cho.

###  xác suất chuyển trạng thái trong một mô hình Hidden Markov Model (HMM).
P(NN|JJ) = 1/4. P(VB|JJ) = 1/6, and P(JJ|JJ) = 1/3.

Tổng các xác suất chuyển từ trạng thái JJ sang các trạng thái khác phải bằng 1, vì trong một mô hình HMM, tổng xác suất chuyển từ bất kỳ trạng thái nào phải luôn bằng 1.

P(NN|JJ) + P(VB|JJ) + P(JJ|JJ) + P(RB|JJ) = 1

### CRF (Conditional Random Field) và HMM (Hidden Markov Model)
CRF (Conditional Random Fields) và HMM (Hidden Markov Models) là hai mô hình thống kê được sử dụng phổ biến trong xử lý chuỗi (sequence modeling), đặc biệt trong các bài toán như nhận dạng thực thể có tên (NER), gán nhãn phần tử (POS tagging) và dịch máy.

- HMM: Giả thuyết Markov có nghĩa là trạng thái ẩn tại thời điểm t chỉ phụ thuộc vào trạng thái tại thời điểm t−1, và các quan sát tại thời điểm t chỉ phụ thuộc vào trạng thái tại thời điểm t. Điều này dẫn đến việc HMM có khả năng mô hình hóa các mối quan hệ ngắn hạn giữa các trạng thái.
- CRF: Không có giả thuyết Markov. CRF có thể sử dụng các đặc trưng long-range (dài hạn) giữa các quan sát, cho phép mô hình này ghi nhận mối quan hệ giữa các yếu tố trong một chuỗi quan sát lớn hơn, giúp giải quyết tốt hơn các vấn đề có mối quan hệ lâu dài.

- HMM: Phù hợp với các bài toán mà quan sát tại mỗi thời điểm phụ thuộc vào một trạng thái ẩn cụ thể. Các ứng dụng điển hình bao gồm nhận dạng giọng nói, dịch máy, và phân tích chuỗi thời gian.
- CRF: CRF thường được sử dụng trong các bài toán nhãn chuỗi (sequence labeling), đặc biệt là trong xử lý ngôn ngữ tự nhiên (NLP), chẳng hạn như phân loại từ loại (POS tagging), nhận dạng thực thể có tên (NER), gán nhãn từ (word segmentation) và phân tích cú pháp (syntactic parsing).

### Finite Automata (FA)
Finite Automata (FA), hay Máy tự động hữu hạn, là một mô hình toán học dùng để mô tả các hệ thống có trạng thái hữu hạn. FA được sử dụng rộng rãi trong lý thuyết tính toán, xử lý ngôn ngữ hình thức, và xây dựng trình biên dịch.

Một Finite Automata thường được định nghĩa bởi một ngũ giác M=(Q,Σ,δ,q0,F), trong đó:
- Q: Tập hợp hữu hạn các trạng thái.
- Σ: Bảng chữ cái đầu vào (tập hợp các ký hiệu đầu vào).
- δ: Hàm chuyển trạng thái (δ:Q×Σ→Q), chỉ định trạng thái tiếp theo dựa trên trạng thái hiện tại và ký hiệu đầu vào.
- q0: Trạng thái bắt đầu, q0∈Q.
- F: Tập các trạng thái kết thúc (trạng thái chấp nhận), F⊆Q.

Finite Automata được chia thành hai loại chính:

- Deterministic Finite Automaton (DFA): Mỗi trạng thái và ký hiệu đầu vào chỉ dẫn đến một trạng thái duy nhất.
- Non-deterministic Finite Automaton (NFA): Có thể có nhiều chuyển trạng thái từ một trạng thái với cùng một ký hiệu đầu vào.

- In NFA, null (or ε) move is allowed i.e., it can move forward without reading symbols.

### Finite-State Transducers (FS Transducers, FST) 
là một mô hình toán học mở rộng của Finite Automata

- Deterministic FST (DFST): Với mỗi trạng thái và ký hiệu đầu vào, luôn có duy nhất một trạng thái tiếp theo và một chuỗi đầu ra.
- Non-deterministic FST (NFST): Có thể tồn tại nhiều trạng thái tiếp theo và chuỗi đầu ra cho một trạng thái và ký hiệu đầu vào cụ thể.

Tùy vào mục đích sử dụng, FST có thể được cấu hình để hoạt động theo các chế độ như generation, recognition, translation, hoặc analysis:

- Generation Mode: Sinh đầu ra từ đầu vào dựa trên các quy tắc đã định nghĩa. VD: Một FST nhận vào dạng cơ bản của từ trong tiếng Anh như walk + past và sinh ra từ dạng chia thì: walked.
- Recognition Mode: Kiểm tra xem một chuỗi đầu vào có được chấp nhận bởi FST hay không. VD: Kiểm tra xem từ "running" có phải là một từ hợp lệ trong ngôn ngữ tiếng Anh hay không.
- Translation Mode: Biến đổi một chuỗi đầu vào thành một chuỗi đầu ra với các quy tắc chuyển đổi phức tạp. Vd: Chuyển đổi từ tiếng Anh cat thành từ tương đương trong tiếng Pháp chat.
- Analysis Mode: Phân tích chuỗi đầu vào và ánh xạ nó về dạng cơ bản hoặc biểu diễn trừu tượng. VD: Từ đầu vào cats, FST tạo ra đầu ra cat + PLURAL.

### Orthographic rules: Finite State Transducer (FST)

### Zip's Law
Zipf's Law nói rằng trong một ngữ cảnh cụ thể (ví dụ như một corpus văn bản), tần suất của từ f và vị trí của nó trong bảng xếp hạng từ (theo tần suất giảm dần) r có mối quan hệ gần đúng như sau:

![image](https://github.com/user-attachments/assets/03da6089-2578-4446-a21d-c587c7cba45b)

![image](https://github.com/user-attachments/assets/36a5947a-9c96-4271-9f53-99468515a728)

-------------------------------------------------------------------------
### POS tagging
Parts-of-Speech (POS) Tagging (Gắn nhãn thành phần của câu) là một nhiệm vụ trong xử lý ngôn ngữ tự nhiên (NLP), trong đó mỗi từ trong một câu hoặc đoạn văn được gắn nhãn với một loại từ (part of speech) phù hợp. Các loại từ này bao gồm danh từ (noun), động từ (verb), tính từ (adjective), trạng từ (adverb), đại từ (pronoun), giới từ (preposition), liên từ (conjunction), và các loại từ khác.

POS tagging giúp hệ thống hiểu được vai trò ngữ pháp của mỗi từ trong câu, từ đó giúp nâng cao khả năng phân tích và xử lý ngữ nghĩa của văn bản. Việc xác định loại từ của mỗi từ trong một câu là cơ sở quan trọng cho nhiều tác vụ NLP khác như phân tích cú pháp, phân loại văn bản, dự đoán từ tiếp theo, hoặc dịch máy.

### Parsing (Phân tích cú pháp):
Parsing trong xử lý ngôn ngữ tự nhiên (NLP) là quá trình phân tích cú pháp của một câu, tức là xác định cấu trúc ngữ pháp của câu dựa trên các quy tắc ngữ pháp của ngôn ngữ đó.
Mục tiêu của parsing là xây dựng một cây cú pháp (parse tree) hoặc cây phân tích cú pháp (syntax tree), trong đó các từ được tổ chức theo một cách thể hiện các quan hệ cú pháp của chúng (ví dụ: chủ ngữ, động từ, bổ ngữ).

### Fuzzy Logic (Lý thuyết mờ) 
là một phương pháp toán học được phát triển để xử lý các vấn đề trong đó thông tin không rõ ràng, không chính xác hoặc không hoàn toàn. Lý thuyết mờ là sự mở rộng của logic cổ điển (Boolean logic), nơi các giá trị có thể là 1 (đúng) hoặc 0 (sai). Trong fuzzy logic, thay vì chỉ có hai giá trị, các sự kiện có thể có giá trị thuộc dải mờ từ 0 đến 1, thể hiện mức độ đúng (hoặc mức độ tin cậy) của một sự kiện.

### Text classification model
Trình tự trong text classification model:
1. Text cleaning:
The goal of text cleaning is to standardize and preprocess the text in a way that the model can more easily understand. Common techniques in text cleaning include:
- Removing or replacing special characters and punctuation.
- Converting all text to lowercase.
- Removing stop words (optional, depending on the model).
- Correcting spelling errors.
- Tokenizing text (splitting it into words or subwords).
- Stemming or lemmatization (reducing words to their base form).

2. Text annotation:
Annotation is the process of labeling or categorizing the text data with the correct class or category labels (in supervised learning). For example, in sentiment analysis, each document might be labeled as "positive", "negative", or "neutral".

3. Text to predictors (Feature Extraction)
Convert the text into a numerical format that can be fed into a machine learning model. This process is called feature extraction or text vectorization.
You convert text into numerical features (predictors) using methods such as:
- Bag of Words (BoW): Representing text as a matrix of word occurrences.
- TF-IDF (Term Frequency-Inverse Document Frequency): A statistical measure that evaluates how important a word is in a document relative to the whole dataset.
- Word Embeddings: Using pre-trained models like Word2Vec or GloVe to convert words into dense vectors.

4. Gradient descent
Train the model

5. Model tuning
Model tuning (also called hyperparameter tuning) is the process of adjusting the model’s hyperparameters to improve performance.

### Flexible Text Matching
Flexible Text Matching (Khớp văn bản linh hoạt) là một kỹ thuật được sử dụng để so sánh và tìm kiếm văn bản trong các ứng dụng xử lý ngôn ngữ tự nhiên (NLP) hoặc tìm kiếm thông tin. Flexible text matching cho phép đối phó với các biến thể trong văn bản như lỗi chính tả, sự thay đổi thứ tự từ, viết tắt, hay sự khác biệt ngữ nghĩa.

Các tasks:
- Information Retrieval (Tìm kiếm thông tin)
- Question Answering (QA)
- Semantic Search (Tìm kiếm ngữ nghĩa)
- Text Classification and Labeling (Phân loại và gán nhãn văn bản)
- Text Summarization (Tóm tắt văn bản)
- Entity Recognition and Linkage (Nhận diện và liên kết thực thể)
- Sentiment Analysis (Phân tích cảm xúc)
- Plagiarism Detection (Phát hiện đạo văn)
- Fuzzy Matching (Khớp mờ)
- Text Mining (Khai thác văn bản)

Đặc điểm:
- Chấp nhận sai sót chính tả: Khả năng nhận diện và xử lý các lỗi chính tả hoặc từ ngữ không chính xác mà người dùng có thể nhập vào trong khi tìm kiếm. Ví dụ: Tìm kiếm "computor" sẽ trả kết quả giống như "computer".
- Xử lý từ đồng nghĩa: Kỹ thuật này có thể nhận diện các từ khác nhau nhưng mang cùng một ý nghĩa. Ví dụ: "car" và "automobile" có thể được coi là tương đương trong việc tìm kiếm.
- Xử lý biến thể từ (stemming/lemmatization): Flexible text matching có thể nhận diện các dạng biến thể của một từ, ví dụ như từ "running", "runner", "ran" có thể được quy về gốc từ "run".
  + Stemming: Là phương pháp làm giảm từ về dạng gốc của nó (thường không phải là một từ có nghĩa hoàn chỉnh).
  + Lemmatization: Là một phương pháp tương tự, nhưng thường đảm bảo rằng từ kết quả là một từ có nghĩa hoàn chỉnh (từ gốc của từ trong từ điển).
- Khớp với các cụm từ (Phrase matching): Flexible text matching có thể nhận diện các cụm từ tương tự, ngay cả khi có sự thay đổi nhẹ về cách diễn đạt hoặc thứ tự của từ. Ví dụ: "high performance computing" và "computing for high performance" có thể được coi là tương đương trong việc tìm kiếm.
- Chấp nhận lỗi cú pháp và ngữ pháp: Hệ thống tìm kiếm linh hoạt có thể hiểu được văn bản ngay cả khi có sự thay đổi trong cấu trúc câu, như khi người dùng nhập một câu không chính xác về ngữ pháp.

Các kỹ thuật: 
- Regular Expressions
- Fuzzy Matching:Kỹ thuật này sử dụng các thuật toán như Levenshtein distance (hoặc độ lệch Hamming) để tính toán mức độ tương đồng giữa các chuỗi văn bản, cho phép khớp với các từ hoặc cụm từ có sự khác biệt nhỏ như lỗi chính tả, sự thay đổi vị trí của ký tự, v.v.
- Cosine Similarity
- NLP-based Matching: Các mô hình NLP như Word2Vec, GloVe, hoặc BERT
- Soundex/Metaphone: Đây là các thuật toán chuyển đổi từ thành mã âm thanh, giúp nhận diện các từ có âm thanh tương tự mặc dù chúng có thể viết khác nhau (thường dùng trong trường hợp khớp tên hoặc từ).

### ELMo
ELMo sử dụng một mô hình ngôn ngữ dạng LSTM (Long Short-Term Memory) hai chiều để sinh ra các embedding từ ngữ cảnh, tức là mỗi từ sẽ có một biểu diễn khác nhau tuỳ vào vị trí của nó trong câu. Điều này giúp mô hình hiểu rõ hơn về nghĩa của từ trong ngữ cảnh, thay vì sử dụng một embedding cố định cho từ đó trong mọi tình huống.  
Ví dụ: Từ "bank" trong câu "I went to the bank to deposit money" sẽ có một embedding khác với từ "bank" trong câu "The boat is on the bank of the river", vì nghĩa của từ "bank" thay đổi tuỳ theo ngữ cảnh của câu.

### XLNet
XLNet là một mô hình ngôn ngữ mạnh mẽ trong học sâu (deep learning) được phát triển bởi Google AI, và là một sự cải tiến của BERT (Bidirectional Encoder Representations from Transformers)

XLNet kết hợp cả hai phương pháp autoregressive (dự đoán từ tiếp theo trong câu, ví dụ như GPT) và autoencoding, giúp mô hình không chỉ hiểu ngữ cảnh mà còn có khả năng sinh văn bản dựa trên ngữ cảnh.
- Autoregressive (như GPT): Dự đoán từ tiếp theo trong câu, giúp mô hình học khả năng sinh câu hợp lý.
- Autoencoding (như BERT): Học cách tái tạo văn bản với một phần của từ (các từ bị ẩn) để nắm bắt ngữ nghĩa của câu.

Permutation Language Modeling: Thay vì chỉ dựa vào thứ tự từ trong câu, XLNet hoán vị các từ trong câu trong quá trình huấn luyện để học được mọi sự kết hợp của các từ trong ngữ cảnh. Điều này giúp mô hình nắm bắt được mối quan hệ giữa các từ trong câu một cách linh hoạt và chính xác hơn.

Transformer Architecture: XLNet sử dụng Transformer architecture (kiến trúc Transformer), tương tự như BERT và GPT, bao gồm các lớp self-attention để xử lý các mối quan hệ giữa các từ trong câu mà không cần phải xử lý tuần tự (RNN, LSTM). Việc này giúp tăng tốc quá trình huấn luyện và giúp mô hình học tốt hơn các đặc trưng ngữ nghĩa phức tạp.

### Transformer XL
Transformer-XL (Transformer eXtra Long) là một biến thể của mô hình Transformer, được phát triển để giải quyết các vấn đề liên quan đến việc xử lý các chuỗi văn bản dài trong học sâu (deep learning) và Xử lý Ngôn ngữ Tự nhiên (NLP).

Khó khăn với transformer truyền thống:
- Giới hạn về chiều dài văn bản: Các mô hình Transformer truyền thống, như BERT và GPT, có một hạn chế về độ dài tối đa của chuỗi văn bản mà chúng có thể xử lý.
- Khó khăn trong việc giữ lại thông tin ngữ cảnh dài hạn: Transformer sử dụng cơ chế self-attention để tạo ra các mối quan hệ giữa các từ trong câu, nhưng với các văn bản dài, việc tính toán attention trên toàn bộ văn bản có thể rất tốn kém và mất hiệu quả.

Cải tiến:
- Segment-level Recurrence: là một kỹ thuật giúp duy trì thông tin ngữ cảnh qua nhiều đoạn văn bản. Thay vì chỉ sử dụng thông tin ngữ cảnh trong một chuỗi văn bản cố định (giới hạn về chiều dài), Transformer-XL chia văn bản thành các đoạn nhỏ (segments) và cho phép các đoạn trước đó chia sẻ thông tin với các đoạn sau. Điều này giúp mô hình duy trì ngữ cảnh dài hơn mà không cần phải tính toán attention trên toàn bộ văn bản. Khi xử lý một đoạn mới, mô hình không chỉ xem xét thông tin trong đoạn đó mà còn sử dụng thông tin từ các đoạn trước đó, giúp cải thiện khả năng hiểu ngữ nghĩa trong các văn bản dài.
- Relative Positional Encodings: Transformer-XL thay thế absolute positional encodings (mã hóa vị trí tuyệt đối) bằng relative positional encodings (mã hóa vị trí tương đối). Điều này giúp mô hình hiểu được mối quan hệ vị trí giữa các từ không chỉ trong một đoạn cụ thể mà còn giữa các đoạn khác nhau trong chuỗi văn bản dài, làm cho mô hình hiệu quả hơn khi xử lý ngữ cảnh dài hạn.

-------------------------------------------------------------
## Ngôn ngữ học (bao gồm Ngữ nghĩa và Hình thái học)

### Các thuật ngữ
- Synonyms (Từ đồng nghĩa)
- Hyponyms (Từ hạ nghĩa)
- Meronyms (Từ bộ phận)
- Homonyms (Từ đồng âm)
- anaphora (khi một đại từ tham chiếu lại một danh từ đã được đề cập trước đó)
- cataphora (khi một đại từ tham chiếu tới một danh từ sẽ xuất hiện sau trong văn bản).
- polarity categories: Phân loại tiêu cực: positive, negative, neutral

### Các loại quan hệ trong ngữ nghĩa học từ vựng
1. Hypernym - Hyponym:
- Hypernym (siêu nghĩa) là một từ có nghĩa rộng hơn, bao trùm nhiều đối tượng hoặc khái niệm cụ thể.
- Hyponym (hạ nghĩa) là một từ chỉ một đối tượng, khái niệm cụ thể thuộc phạm vi của từ siêu nghĩa.

2. Meronym - Holonym:
- Meronym là từ chỉ một phần của một đối tượng hoặc tổng thể.
- Holonym là từ chỉ tổng thể mà phần của nó thuộc về.
Ví dụ: "Window" (cửa sổ) là meronym của "room" (phòng), vì cửa sổ là một phần của phòng.

![image](https://github.com/user-attachments/assets/ab07281f-2135-4653-ab82-61f0ca0af393)

### Perplexity 
là một chỉ số đo độ khó của mô hình trong việc dự đoán một chuỗi các từ. Cụ thể, perplexity càng thấp, mô hình càng "chắc chắn" và "dự đoán tốt" hơn về chuỗi từ.

Khi tính perplexity của một mô hình ngôn ngữ không smoothing trên một tập dữ liệu kiểm tra (test corpus) có chứa các từ chưa gặp, ta gặp phải một vấn đề:

Nếu mô hình không xử lý trường hợp từ chưa gặp (ví dụ, không sử dụng kỹ thuật smoothing như Laplace smoothing), xác suất của các từ này sẽ bằng 0.
Khi xác suất của một từ bằng 0, điều này sẽ khiến xác suất của cả chuỗi trở thành 0, và từ đó logarit của xác suất sẽ không xác định (log(0) không tồn tại).
Vì vậy, perplexity sẽ bị vô định hoặc là vô cùng (infinity), do sự hiện diện của các từ chưa gặp.

perplexity = 2^H(T)

H(T) = -1/N * Σ(log2P)

### Symbolic Analysis
- Parsing (Phân tích cú pháp): Quá trình phân tích cú pháp trong symbolic NLP sử dụng các quy tắc hình thức để xây dựng cây cú pháp (parse tree) cho một câu, xác định cấu trúc ngữ pháp của câu đó.
- Semantic interpretation (Diễn giải ngữ nghĩa): Các mô hình symbolic có thể xây dựng các biểu diễn ngữ nghĩa từ các câu văn bằng cách sử dụng các quy tắc để hiểu mối quan hệ giữa các đối tượng trong câu (ví dụ: "John eats an apple" có thể được biểu diễn dưới dạng một biểu thức logic như "eats(John, apple)").
- Knowledge representation: Các hệ thống symbolic có thể sử dụng lý thuyết đồ thị hoặc mạng lưới khái niệm để đại diện cho các thực thể và mối quan hệ giữa chúng trong cơ sở tri thức (knowledge base). Điều này giúp hệ thống có thể suy luận và trả lời các câu hỏi một cách logic.

-------------Hình thái học---------------------------------
### Các loại hình thái học (Morphology)
Morphology (tạm dịch là Hình thái học) là một nhánh của ngôn ngữ học nghiên cứu về cấu trúc và hình thức của từ trong ngôn ngữ, đặc biệt là cách mà từ được hình thành từ các thành phần nhỏ hơn gọi là morphen (hình vị). Các hình vị này là đơn vị ngữ nghĩa cơ bản, không thể chia nhỏ hơn nữa mà vẫn giữ nguyên nghĩa.

1. Inflectional Morphology (Hình thái biến hình): 
Hình thái biến hình liên quan đến các thay đổi hình thái của từ để thể hiện các đặc tính ngữ pháp như số, thì, dạng sở hữu, v.v. Tuy nhiên, hình thái biến hình không thay đổi loại từ và thường chỉ thay đổi cấu trúc ngữ pháp của từ mà không làm thay đổi nghĩa cơ bản của từ đó.
Ví dụ: "dog" (chó) và "dogs" (chó, số nhiều), hay "run" (chạy) và "running" (đang chạy) – loại từ vẫn là danh từ và động từ, chỉ có sự thay đổi về ngữ pháp.

2. Derivational Morphology (Hình thái tạo từ): Hình thái tạo từ thay đổi loại từ và nghĩa của từ. Khi một từ được tạo ra từ một từ gốc bằng cách thêm tiền tố hoặc hậu tố, loại từ và/hoặc nghĩa của từ mới có thể thay đổi đáng kể.

Ví dụ:
- "happy" (hạnh phúc) + "ness" → "happiness" (hạnh phúc, danh từ) – từ "happy" (tính từ) trở thành "happiness" (danh từ).
- "teach" (dạy) + "er" → "teacher" (giáo viên) – từ động từ "teach" trở thành danh từ "teacher".
Đây là loại hình thái học thay đổi loại từ và ảnh hưởng đến nghĩa của từ.
3. Cliticization (Hình thái dính liền)

Hình thái dính liền liên quan đến việc thêm các phần tử ngữ pháp (clitics) vào từ mà không thay đổi loại từ hay nghĩa của từ. Các phần tử này thường không thể đứng độc lập mà phải "dính" vào từ chính để tạo thành một cấu trúc ngữ pháp hợp lý.
Ví dụ: Tiền tố "n't" trong "isn't" (không phải) hay trong "I'll" (tôi sẽ) không thay đổi loại từ hay nghĩa của từ, chỉ có chức năng ngữ pháp.

### Morphological Segmentation
Morphological Segmentation (Phân đoạn hình thái học) trong xử lý ngôn ngữ tự nhiên (NLP) là quá trình phân tách từ ngữ thành các thành phần cơ bản của nó, gọi là morphemes (hình thái học).

Morpheme: Là đơn vị ngữ nghĩa nhỏ nhất trong ngôn ngữ. Ví dụ:

Từ "cats" có thể được phân đoạn thành hai morphemes:
- "cat" (morpheme gốc, có nghĩa là "mèo").
- "s" (morpheme cú pháp, chỉ sự số nhiều).

### Các loại morphemes:
1. Free Morphemes (Morpheme tự do):
Morpheme tự do là những morpheme có thể đứng độc lập và có nghĩa mà không cần phải kết hợp với bất kỳ morpheme nào khác.

Ví dụ: "book", "run", "cat" – tất cả đều có thể là từ hoàn chỉnh và có nghĩa độc lập. Chú ý: Các morphemes này có thể là từ gốc (root words) mà không cần sự kết hợp.
2. Bound Morphemes (Morpheme phụ thuộc):
Morpheme phụ thuộc là những morphemes không thể đứng độc lập mà phải được kết hợp với morpheme khác để tạo thành từ có nghĩa.  
Chúng có thể là hậu tố (suffix), tiền tố (prefix), hay inflectional morphemes (morpheme biến hình).  
Ví dụ:  
- "un-" trong "unhappy" (không vui) – không thể đứng độc lập mà phải đi kèm với "happy".
- "-ed" trong "walked" (đã đi) – không thể tự đứng mà phải đi kèm với "walk".

Bound morphemes có vai trò quan trọng trong việc thay đổi nghĩa hoặc hình thức ngữ pháp của từ, nhưng không thể tồn tại một mình.

3. Derived Morphemes (Morpheme tạo từ):
Morpheme tạo từ là những morphemes mà khi thêm vào một từ gốc, sẽ tạo ra một từ mới với nghĩa mới, hoặc thay đổi loại từ.  
Ví dụ: "-ness" trong "happiness" (hạnh phúc) hay "-ly" trong "quickly" (nhanh chóng).  
Đây có thể là một loại bound morpheme, vì nó thường không thể đứng độc lập mà phải kết hợp với từ gốc.  
4. Lexical Morphemes (Morpheme từ vựng):
Lexical morphemes là những morphemes mang nghĩa cụ thể, thường là từ gốc, và chúng có thể là free morphemes.  
Ví dụ: "book", "cat", "run". Các từ này mang nghĩa cụ thể và có thể là từ gốc trong một câu.

### Speech Segmentation (Phân đoạn lời nói) 
là quá trình chia một chuỗi âm thanh liên tục thành các đơn vị có ý nghĩa, như từ, câu, hoặc đoạn văn. Trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP) và nhận diện tiếng nói (speech recognition), speech segmentation là một bước quan trọng để giúp máy tính hiểu và phân tích lời nói một cách chính xác.

### Temporal Probabilistic Reasoning (Lý luận xác suất theo thời gian) 
là một lĩnh vực trong trí tuệ nhân tạo và học máy (machine learning) nghiên cứu cách thức xử lý và mô hình hóa các sự kiện có tính chất thời gian và không chắc chắn. Cụ thể, nó liên quan đến việc áp dụng các mô hình xác suất để mô hình hóa sự thay đổi của các sự kiện theo thời gian và ra quyết định dựa trên các thông tin không hoàn chỉnh hoặc không chắc chắn về tương lai.

### Các loại Sequential Labeling:
1. Part-of-Speech (POS) Tagging:
Mỗi từ trong câu được gán nhãn tương ứng với loại từ của nó (danh từ, động từ, tính từ, trạng từ, v.v.). Ví dụ:
- "I" → PRP (pronoun)
- "like" → VB (verb)
- "cats" → NNS (plural noun)
2. Named Entity Recognition (NER):
Xác định và gán nhãn cho các thực thể có tên trong văn bản, chẳng hạn như tên người, tổ chức, địa điểm, thời gian, v.v. Ví dụ:
- "Apple" → ORGANIZATION
- "Paris" → LOCATION
- "July 2023" → DATE
3. Chunking:
Phân chia câu thành các cụm từ (chunks) và gán nhãn cho mỗi cụm. Ví dụ, cụm danh từ (NP), cụm động từ (VP), v.v.:
- "The quick brown fox" → NP (noun phrase)
- "jumps over the lazy dog" → VP (verb phrase)
4. Dependency Parsing:
Xác định các mối quan hệ cú pháp giữa các từ trong câu, ví dụ như chủ ngữ, vị ngữ, đối tượng, v.v. Đây là một dạng gán nhãn chuỗi trong ngữ cảnh phân tích cú pháp.
5. Bio-Tagging (Bio-Chunking):
Sử dụng hệ thống nhãn BIO (Beginning, Inside, Outside) để gán nhãn các thực thể trong văn bản. Hệ thống này đặc biệt hữu ích trong Named Entity Recognition. Ví dụ:
- "Barack Obama" → B-PER (Beginning of Person) I-PER (Inside Person)
- "in New York" → O (Outside any entity)
6. Sequence Labeling for Sentiment Analysis:
Gán nhãn cho từng từ hoặc cụm từ để xác định cảm xúc (tích cực, tiêu cực, trung lập). Ví dụ:
- "I love this movie!" → positive
- "This movie is terrible." → negative
7. Word Segmentation and Lemmatization:
- Word Segmentation: Chia văn bản thành các từ riêng biệt (đặc biệt trong các ngôn ngữ không có dấu cách giữa các từ như tiếng Trung Quốc, tiếng Nhật, hoặc tiếng Việt).
- Lemmatization: Gán nhãn cho các từ theo dạng gốc của chúng (ví dụ, "running" → "run").

Notes: Types of sequence labeling: token và span

### Các mô hình sử dụng trong Sequential Labeling:
- Hidden Markov Models (HMM):
Một mô hình chuỗi thời gian, thường được sử dụng trong POS tagging và NER. HMM xác định trạng thái (nhãn) của mỗi phần tử dựa trên xác suất có điều kiện và các trạng thái trước đó trong chuỗi.
- Conditional Random Fields (CRF):
Là mô hình mạnh mẽ hơn HMM, CRF là mô hình chuỗi phân loại có điều kiện, nơi các nhãn không chỉ phụ thuộc vào trạng thái trước đó mà còn phụ thuộc vào các yếu tố trong toàn bộ chuỗi.
- Recurrent Neural Networks (RNN) và Long Short-Term Memory (LSTM):
Các mô hình học sâu này có khả năng xử lý chuỗi dài và rất phổ biến trong các bài toán gán nhãn chuỗi, bao gồm POS tagging, NER, và chunking.
Transformers (BERT, GPT, T5):
- Các mô hình dựa trên kiến trúc transformer hiện đại, như BERT (Bidirectional Encoder Representations from Transformers), có thể sử dụng để thực hiện gán nhãn chuỗi trong các tác vụ như NER, POS tagging, và sentiment analysis. BERT đặc biệt mạnh mẽ vì có thể xử lý mối quan hệ giữa các từ trong câu một cách toàn diện và hiệu quả.

-------------Ngữ nghĩa---------------------------------
### Word Sense Disambiguation (WSD) (Giải nghĩa từ ngữ) 
là một nhiệm vụ trong xử lý ngôn ngữ tự nhiên (NLP) nhằm xác định nghĩa chính xác của một từ trong ngữ cảnh cụ thể của một câu hoặc đoạn văn. Nhiều từ trong ngôn ngữ tự nhiên có đa nghĩa (polysemy), tức là chúng có thể mang nhiều nghĩa khác nhau tùy vào ngữ cảnh sử dụng. WSD là quá trình phân biệt các nghĩa khác nhau của một từ và xác định nghĩa nào phù hợp nhất với ngữ cảnh trong câu.

### Maximum Matching Algorithm (Greedy Algorithm - Forward Pass)
Maximum Matching là một thuật toán tham lam (Greedy) được sử dụng để phân tách chuỗi đầu vào thành các từ có nghĩa (hay còn gọi là phân tích từ vựng) bằng cách tìm từ dài nhất (hoặc từ có nghĩa nhất) trong từ điển và khớp với phần đầu của chuỗi, sau đó tiếp tục làm như vậy với phần còn lại của chuỗi.

### Tóm tắt các loại ambiguity:
- Lexical Ambiguity: Mơ hồ từ vựng (từ có nhiều nghĩa).
- Syntactic Ambiguity: Mơ hồ cú pháp (cấu trúc câu có thể có nhiều cách giải thích).
- Semantic Ambiguity: Mơ hồ nghĩa (một câu hoặc cụm từ có nhiều cách hiểu).
- Pragmatic Ambiguity: Mơ hồ thực dụng (ngữ cảnh giao tiếp quyết định nghĩa của câu).
- Structural Ambiguity: Mơ hồ cấu trúc (câu có thể phân tích theo nhiều cách).
- Referential Ambiguity: Mơ hồ tham chiếu (đại từ có thể tham chiếu đến nhiều thực thể).
- Anaphoric Ambiguity: Mơ hồ anaphora (đại từ tham chiếu lại một thực thể trước đó).
- Hyponymic Ambiguity: Mơ hồ quan hệ hyponym-hypernym (từ có thể đại diện cho nhiều loài con của một nhóm).
- Phonological ambiguity: (mơ hồ ngữ âm) là một loại mơ hồ xảy ra khi một từ hoặc cụm từ có cách phát âm giống nhau (hoặc gần giống nhau) nhưng có nghĩa khác nhau.
- Discourse ambiguities (mơ hồ trong diễn ngôn) đề cập đến những sự không rõ ràng hoặc mơ hồ trong việc hiểu và giải thích mối quan hệ giữa các câu hoặc các phần trong một đoạn văn hoặc văn bản, gây khó khăn trong việc xác định nghĩa chính xác của một thông điệp.
- Scope ambiguity (mơ hồ phạm vi) xảy ra khi không rõ ràng phạm vi của một từ hoặc cụm từ trong câu, dẫn đến việc hiểu sai ý nghĩa của câu. Mơ hồ phạm vi thường gặp với các từ phủ định, lượng từ, động từ hoặc các cụm từ có thể áp dụng cho một phần của câu hoặc cho toàn bộ câu, gây khó khăn trong việc xác định đúng ngữ nghĩa của câu.

### Coreference Resolution
Coreference Resolution (Giải quyết tham chiếu) là một quá trình trong xử lý ngôn ngữ tự nhiên (NLP) nhằm xác định các từ hoặc cụm từ trong văn bản mà chúng đều ám chỉ cùng một đối tượng hoặc thực thể.
Ví dụ, trong câu:

"John went to the store. He bought some milk."

Ở đây, "John" và "He" tham chiếu đến cùng một người, vì vậy nhiệm vụ của coreference resolution là nhận diện rằng "He" trong câu thứ hai thực sự là "John" trong câu đầu tiên.

Các bước trong Coreference Resolution:
- Nhận diện các thực thể (Entity Recognition): Xác định các danh từ hoặc cụm danh từ trong văn bản (ví dụ: "John", "the store").
- Tìm các liên kết tham chiếu (Coreferent Mentions): Tìm các phần của văn bản (thường là các đại từ hoặc danh từ khác) mà chúng tham chiếu đến những thực thể đã được nhận diện trước đó.
- Xác định nhóm tham chiếu (Coreference Chains): Các thực thể và đại từ liên quan sẽ được nhóm lại thành một chuỗi tham chiếu (coreference chain). Ví dụ: trong văn bản trên, chuỗi tham chiếu có thể là: "John" → "He".

### Co-reference và Anaphora
Co-reference là một khái niệm rộng hơn, chỉ sự liên hệ giữa hai hay nhiều biểu thức trong một câu hoặc đoạn văn mà chúng tham chiếu đến cùng một thực thể hoặc đối tượng. Khi hai từ hoặc cụm từ có quan hệ co-reference, chúng đều chỉ đến cùng một thực thể trong thế giới thực hoặc trong ngữ cảnh văn bản. Anaphora là một dạng của co-reference, nhưng không phải tất cả các mối quan hệ co-reference đều là anaphora.

Ví dụ về co-reference:
- "John and Mary are friends. They like playing tennis."

Ở đây, "They" tham chiếu đến cả "John" và "Mary". Đây là một ví dụ của co-reference giữa hai đối tượng.

- "The president gave a speech yesterday. The speech was very inspiring."

Trong trường hợp này, "the speech" và "the president" có quan hệ co-reference, nhưng không phải là anaphora, vì "the speech" không phải là đại từ thay thế cho một danh từ đã được nhắc đến trước đó mà là sự tham chiếu đến sự kiện đã được mô tả.

### Deixis
Deixis là một khái niệm trong ngữ nghĩa học và lý thuyết ngữ dụng học, dùng để chỉ những từ ngữ hoặc biểu thức có ý nghĩa phụ thuộc vào ngữ cảnh giao tiếp, đặc biệt là các yếu tố như thời gian, không gian, người nói và người nghe. Những từ này không thể được hiểu rõ ràng chỉ từ chính bản thân chúng mà phải xét đến ngữ cảnh cụ thể trong đó chúng được sử dụng.

Types:
- Person Deixis
- Temporal Deixis (Deixis về thời gian)
- Spatial Deixis (Deixis về không gian)
- Discourse Deixis (Deixis về diễn ngữ)

### Implicature (ngụ ý) 
là một khái niệm trong lý thuyết ngữ dụng học, được giới thiệu bởi nhà ngữ học Grice, dùng để chỉ những thông tin mà người nói ám chỉ hoặc ngụ ý trong một cuộc trò chuyện mà không nói rõ ra. Các ngụ ý này không phải là những gì được nói trực tiếp trong câu, mà là những điều người nghe có thể hiểu hoặc suy luận từ ngữ cảnh, từ các quy tắc ngữ dụng, hoặc từ những gì được nói trước đó.

Types:
- Conversational Implicature (Ngụ ý trong giao tiếp): Đây là ngụ ý được tạo ra trong quá trình giao tiếp, dựa trên các nguyên tắc giao tiếp mà người nói và người nghe cùng hiểu. Ngụ ý này không phải là một phần của nghĩa từ vựng của câu mà người nghe có thể suy luận từ ngữ cảnh và các quy tắc giao tiếp thông thường
- Conventional Implicature (Ngụ ý quy ước): Đây là ngụ ý mà người nghe có thể rút ra từ nghĩa của từ hoặc cấu trúc câu, nhưng ngụ ý này không phụ thuộc vào ngữ cảnh. Các ngụ ý này tồn tại vì các từ hoặc biểu thức mang chúng theo một cách thức quy ước.
- Unconversational Implicature

### Presupposition (giả định) 
là một khái niệm trong lý thuyết ngữ dụng học và ngữ nghĩa học, đề cập đến những thông tin hoặc giả định mà người nói "giả định" là đã được người nghe chấp nhận là đúng trước khi họ nói điều gì đó. Những giả định này là nền tảng mà cuộc trò chuyện hoặc câu nói được xây dựng trên đó. Presupposition khác với implicature (ngụ ý) vì giả định là những thông tin mà nếu không có chúng, câu nói sẽ trở nên vô nghĩa hoặc không hợp lý trong ngữ cảnh giao tiếp.

### Speech act (hành động lời nói) 
là một khái niệm quan trọng trong lý thuyết ngữ dụng học (pragmatics), đề cập đến hành động mà người nói thực hiện khi họ phát biểu một câu trong giao tiếp. Mỗi câu trong giao tiếp không chỉ đơn giản là một tuyên bố (statement), mà còn là một hành động, ví dụ như yêu cầu, hứa hẹn, mời gọi, hay xin lỗi.

Types:
- Representatives
- Comisessives
- Directives
- Declarations
- Expressives
