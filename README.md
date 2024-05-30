# Graph Embedding for Social Network Analysis

## Description

This project focuses on utilizing topological Node2vec for enhanced graph embedding to analyze social networks. The primary objectives are to detect communities, predict links, and gain insights into the structure of social networks. This approach leverages advanced techniques in graph embedding, persistent homology, community detection, and link prediction.

## Skills Demonstrated

- **Graph Embedding:** Techniques to represent nodes of a graph in a low-dimensional space.
- **Persistent Homology:** Methods to study the topological features of data that persist across multiple scales.
- **Community Detection:** Algorithms to identify clusters or communities within a network.
- **Link Prediction:** Techniques to predict the likelihood of future or missing connections between nodes.

## Use Case

- **Social Media Analytics:** Analyzing user interactions and behaviors to derive insights and trends.
- **Recommendation Systems:** Providing personalized recommendations based on the network structure.
- **Fraud Detection:** Identifying suspicious patterns and potential fraudulent activities within a network.

## Components

### 1. Data Collection and Preprocessing

Collect and preprocess social network data to ensure it is clean, consistent, and ready for analysis.

- **Data Sources:** Social media platforms, online forums, interaction logs.
- **Techniques Used:** Data cleaning, normalization, feature extraction, handling missing data.

### 2. Graph Embedding with Node2vec

Utilize Node2vec to generate embeddings that capture the topological structure of the graph.

- **Techniques Used:** Random walks, Skip-gram model.
- **Libraries/Tools:** Node2vec, Gensim.

### 3. Persistent Homology

Apply persistent homology to study the topological features of the graph that are invariant under continuous transformations.

- **Techniques Used:** Filtration, Betti numbers, Persistence diagrams.
- **Libraries/Tools:** GUDHI, Ripser.

### 4. Community Detection

Implement community detection algorithms to identify clusters or groups within the social network.

- **Techniques Used:** Modularity optimization, Spectral clustering.
- **Algorithms Used:** Louvain method, Girvan-Newman algorithm.

### 5. Link Prediction

Develop link prediction models to estimate the likelihood of future or missing connections in the graph.

- **Techniques Used:** Similarity measures, Machine learning models.
- **Algorithms Used:** Logistic Regression, Random Forest, Graph Neural Networks.

### 6. Evaluation and Validation

Evaluate the performance of the graph embedding, community detection, and link prediction models using appropriate metrics.

- **Metrics Used:** AUC-ROC, Precision, Recall, F1-score, Modularity.

## Project Structure

```
graph_embedding_social_network_analysis/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── graph_embedding.ipynb
│   ├── persistent_homology.ipynb
│   ├── community_detection.ipynb
│   ├── link_prediction.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── graph_embedding.py
│   ├── persistent_homology.py
│   ├── community_detection.py
│   ├── link_prediction.py
│   ├── evaluation.py
├── models/
│   ├── graph_embedding_model.pkl
│   ├── link_prediction_model.pkl
├── README.md
├── requirements.txt
├── setup.py
```

## Getting Started

### Prerequisites

- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graph_embedding_social_network_analysis.git
   cd graph_embedding_social_network_analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place raw social network data files in the `data/raw/` directory.
2. Run the data preprocessing script to prepare the data:
   ```bash
   python src/data_preprocessing.py
   ```

### Running the Notebooks

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open and run the notebooks in the `notebooks/` directory to preprocess data, develop models, and evaluate the system:
   - `data_preprocessing.ipynb`
   - `graph_embedding.ipynb`
   - `persistent_homology.ipynb`
   - `community_detection.ipynb`
   - `link_prediction.ipynb`
   - `evaluation.ipynb`

### Training and Evaluation

1. Train the graph embedding models:
   ```bash
   python src/graph_embedding.py --train
   ```

2. Train the link prediction models:
   ```bash
   python src/link_prediction.py --train
   ```

3. Evaluate the models:
   ```bash
   python src/evaluation.py --evaluate
   ```

## Results and Evaluation

- **Graph Embedding:** Successfully generated embeddings that capture the topological structure of the social network.
- **Community Detection:** Identified clusters within the network with high modularity scores.
- **Link Prediction:** Developed models that accurately predict future or missing links with high precision and recall.

## Contributing

We welcome contributions from the community. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and supporters of this project.
- Special thanks to the graph theory and machine learning communities for their invaluable resources and support.
```
