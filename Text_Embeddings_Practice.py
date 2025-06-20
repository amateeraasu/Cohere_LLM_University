#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary tools
import cohere
import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import requests
from io import StringIO


# In[2]:


# Connecting to your API KEY in Cohere
co = cohere.ClientV2("---")


# In[3]:


# We will Load a news dataset. We'll create a sample dataset for demonstration.
#    In practice, you can use datasets like:
#    - BBC News Dataset
#    - AG News Dataset  
#    - Reuters Dataset
   

def load_news_dataset():
    sample_data={
        'category': [
            'tech', 'tech', 'tech', 'tech', 'tech',
            'sports', 'sports', 'sports', 'sports', 'sports',
            'business', 'business', 'business', 'business', 'business',
            'politics', 'politics', 'politics', 'politics', 'politics',
            'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment' ],
        'headline': [
            'Apple releases new iPhone with advanced AI features',
            'Google announces breakthrough in quantum computing',
            'Microsoft launches new cloud computing platform',
            'Tesla develops revolutionary battery technology',
            'OpenAI introduces next-generation language model',
            'World Cup final breaks viewership records worldwide',
            'Olympic swimmer sets new world record in Tokyo',
            'NBA playoffs reach thrilling conclusion in finals',
            'Tennis championship sees unexpected upset victory',
            'Soccer transfer window brings major player moves',
            'Stock market reaches all-time high amid growth',
            'Major bank announces merger with competitor',
            'Cryptocurrency prices surge following regulations',
            'Retail giant reports record quarterly earnings',
            'Housing market shows signs of cooling down',
            'Presidential election campaign enters final phase',
            'Congress passes landmark infrastructure bill',
            'Supreme Court delivers major constitutional ruling',
            'International trade negotiations reach agreement',
            'Local government implements new policy reforms',
            'Hollywood blockbuster dominates box office charts',
            'Music festival announces star-studded lineup',
            'Award ceremony celebrates outstanding performances',
            'Streaming service launches exclusive content series',
            'Celebrity couple announces surprise engagement news' ] 
    }
    return pd.DataFrame(sample_data)


# In[4]:


# Loading the dataset
df_orig = load_news_dataset()
print("Original dataset shape:", df_orig.shape)
print("\nCategories:", df_orig['category'].unique())
print("\nSample headlines:")
for i, headline in enumerate(df_orig['headline'].head()):
    print(f"{i+1}. {headline}")


# In[5]:


# Step 4: Sample and prepare data (similar to ATIS approach)
# For this tutorial, we'll use all data, but you can sample like this:
# df = df_orig.sample(frac=0.8, random_state=42)


# In[6]:


# Let's select specific categories for focused analysis
selected_categories = ['tech', 'sports', 'business']
df = df_orig[df_orig.category.isin(selected_categories)].copy()


# In[7]:


# Save categories for later visualization
categories = df['category'].copy()


# In[8]:


# Prepare text data (remove category column for embedding)
text_df = df.drop(columns=['category']).copy()
text_df.reset_index(drop=True, inplace=True)


# In[9]:


print(f"\nFiltered dataset shape: {text_df.shape}")
print("\nSample headlines for embedding:")
for headline in text_df['headline'].head():
    print(f"- {headline}")


# In[10]:


# Step 5: Create embeddings function
def get_embeddings(texts, model="embed-v4.0", input_type="search_document"):
  
    try:
        output = co.embed(
            texts=texts, # texts: List of text strings
            model=model, #  model: Cohere embedding model to use
            input_type=input_type, #input_type: Type of input for optimization
            embedding_types=["float"]
        )
        return output.embeddings.float_ #  List of embedding vectors
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


# In[11]:


# Step 6: Generate embeddings for headlines
print("\nGenerating embeddings...")
text_df['headline_embeds'] = get_embeddings(text_df['headline'].tolist())


# In[12]:


# Check if embeddings were generated successfully
if text_df['headline_embeds'].iloc[0] is not None:
    embedding_dim = len(text_df['headline_embeds'].iloc[0])
    print(f"Successfully generated embeddings with {embedding_dim} dimensions")
else:
    print("Failed to generate embeddings. Check your API key and connection.")


# In[13]:


# Step 7: Visualization functions (adapted from ATIS tutorial)
def get_pc(arr, n):
    """Reduce dimensionality using Principal Component Analysis"""
    pca = PCA(n_components=n)
    embeds_transform = pca.fit_transform(arr)
    return embeds_transform


# In[14]:


def generate_chart(df, xcol, ycol, color_col=None, title=''):
    """Generate 2D scatter plot for embeddings"""
    if color_col:
        chart = alt.Chart(df).mark_circle(size=200).encode(
            x=alt.X(xcol, scale=alt.Scale(zero=False), 
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y=alt.Y(ycol, scale=alt.Scale(zero=False), 
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            color=alt.Color(color_col, 
                          scale=alt.Scale(scheme='category10'),
                          legend=alt.Legend(title="Category")),
            tooltip=['headline', color_col]
        )
    else:
        chart = alt.Chart(df).mark_circle(size=200).encode(
            x=alt.X(xcol, scale=alt.Scale(zero=False)),
            y=alt.Y(ycol, scale=alt.Scale(zero=False)),
            color=alt.value('#333293'),
            tooltip=['headline']
        )
    
    return chart.configure(background="#FDF7F0").properties(
        width=800, height=500, title=title
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14)


# In[15]:


# Step 8: Create visualizations
print("\nCreating visualizations...")
# Convert embeddings to numpy array
embeds = np.array(text_df['headline_embeds'].tolist())


# In[16]:


# ============================================================================
# VISUALIZATION 1: HEATMAP (10 dimensions) - Like ATIS Step 3
# ============================================================================


# In[17]:



# Reduce embeddings to 10 principal components to aid visualization
embeds_pc10 = get_pc(embeds, 10)

# Set sample size to visualize (like ATIS)
sample = min(9, len(text_df))  # Using 9 like in ATIS example

# Reshape the data for visualization purposes (exact ATIS approach)
source = pd.DataFrame(embeds_pc10)[:sample]
source = pd.concat([source, text_df['headline'][:sample]], axis=1)
source = source.melt(id_vars=['headline'])


# In[18]:


# Configure the plot (same style as ATIS)
heatmap_chart = alt.Chart(source).mark_rect().encode(
    x=alt.X('variable:N', title="Embedding"),
    y=alt.Y('headline:N', title='', axis=alt.Axis(labelLimit=500)),
    color=alt.Color('value:Q', title="Value", scale=alt.Scale(
                range=["yellow", "#000000"]))
).configure(background='#ffffff'
        ).properties(
        width=700,
        height=400,
        title='News Headlines Embeddings with 10 dimensions'
       ).configure_axis(
      labelFontSize=15,
      titleFontSize=12)

# Show the heatmap
print("📊 Displaying Heatmap Visualization:")
print("This shows the first 10 principal components of your embeddings")
print("Similar colors indicate similar embedding values\n")

# Display the chart
heatmap_chart


# In[19]:



# ============================================================================
# VISUALIZATION 2: 2D SCATTER PLOT - Like ATIS Step 4  
# ============================================================================


# In[20]:



# Reduce embeddings to 2 principal components to aid visualization
embeds_pc2 = get_pc(embeds, 2)

# Add the principal components to dataframe (ATIS style)
df_pc2 = pd.concat([text_df, pd.DataFrame(embeds_pc2)], axis=1)

# Make sure column names are strings (like ATIS)
df_pc2.columns = df_pc2.columns.astype(str)

# Add categories back for coloring
df_pc2['category'] = categories.reset_index(drop=True)

# Create the 2D scatter plot using the same generate_chart function from ATIS
scatter_chart_basic = generate_chart(
    df_pc2.iloc[:sample], '0', '1', 
    title='2D Embeddings - News Headlines'
)

print("🎯 Displaying 2D Scatter Plot:")
print("Points closer together represent semantically similar headlines")
print("You should see clusters forming by topic!\n")


# In[21]:


# Enhanced version with category colors
def generate_enhanced_chart(df, xcol, ycol, color_col=None, lbl='on', title=''):
    """Enhanced version of generate_chart with category coloring"""
    if color_col:
        chart = alt.Chart(df).mark_circle(size=500).encode(
            x=alt.X(xcol, scale=alt.Scale(zero=False),
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y=alt.Y(ycol, scale=alt.Scale(zero=False),
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            color=alt.Color(color_col, 
                          scale=alt.Scale(scheme='category10'),
                          legend=alt.Legend(title="Category")),
            tooltip=['headline', 'category']
        )
    else:
        chart = alt.Chart(df).mark_circle(size=500).encode(
            x=alt.X(xcol, scale=alt.Scale(zero=False),
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y=alt.Y(ycol, scale=alt.Scale(zero=False),
                   axis=alt.Axis(labels=False, ticks=False, domain=False)),
            color=alt.value('#333293'),
            tooltip=['headline']
        )
    
    if lbl == 'on':
        text = chart.mark_text(
            align='left', baseline='middle', dx=15, size=13, color='black'
        ).encode(text='headline', color=alt.value('black'))
    else:
        text = chart.mark_text(align='left', baseline='middle', dx=10).encode()
        
    result = (chart + text).configure(background="#FDF7F0").properties(
        width=800, height=500, title=title
    ).configure_legend(orient='bottom', titleFontSize=18, labelFontSize=18)
    
    return result


# In[22]:



# Create enhanced 2D visualization with categories
scatter_chart_enhanced = generate_enhanced_chart(
    df_pc2, '0', '1', 'category', lbl='on',
    title='2D Embeddings - Color Coded by Category'
)

print("🌈 Enhanced 2D Plot with Category Colors:")
print("- Tech headlines should cluster together")
print("- Sports headlines should form their own cluster") 
print("- Business headlines should group separately")
print("- Similar topics are closer in the embedding space\n")


# In[23]:


print("=" * 60)
print("📈 VISUALIZATION RESULTS")
print("=" * 60)


# In[24]:



# Show sample headlines being visualized
print(f"\n📝 Headlines being visualized (first {sample}):")
for i, headline in enumerate(text_df['headline'].head(sample), 1):
    category = categories.iloc[i-1] if i-1 < len(categories) else 'unknown'
    print(f"{i}. [{category.upper()}] {headline}")

print(f"\n🔬 EMBEDDING ANALYSIS:")
print(f"- Original embedding dimensions: {embeds.shape[1]}")
print(f"- Reduced to 10D for heatmap")  
print(f"- Reduced to 2D for scatter plot")
print(f"- Total headlines analyzed: {len(text_df)}")
print(f"- Headlines visualized: {sample}")


# In[25]:


heatmap_result = heatmap_chart
scatter_result_basic = scatter_chart_basic  
scatter_result_enhanced = scatter_chart_enhanced


# In[26]:


heatmap_result


# In[27]:


scatter_result_basic


# In[28]:


scatter_result_enhanced 


# In[29]:



# Convert embeddings to numpy array
embeds = np.array(text_df['headline_embeds'].tolist())

# 1. Heatmap visualization (10 dimensions)
embeds_pc10 = get_pc(embeds, 10)
sample_size = min(10, len(text_df))


# In[30]:


# Prepare data for heatmap
heatmap_df = pd.DataFrame(embeds_pc10[:sample_size])
heatmap_df = pd.concat([heatmap_df, text_df['headline'].head(sample_size)], axis=1)
heatmap_df = heatmap_df.melt(id_vars=['headline'])


# In[31]:


# Create heatmap
heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
    x=alt.X('variable:N', title="Embedding Dimension"),
    y=alt.Y('headline:N', title='Headlines', axis=alt.Axis(labelLimit=400)),
    color=alt.Color('value:Q', title="Value", 
                   scale=alt.Scale(range=["#917EF3", "#000000"]))
).configure(background='#ffffff').properties(
    width=700, height=400, title='News Headlines Embeddings (10 dimensions)'
).configure_axis(labelFontSize=12, titleFontSize=14)

print("Heatmap created successfully!")


# In[32]:


# 2. 2D scatter plot
embeds_pc2 = get_pc(embeds, 2)

# Prepare data for 2D plot
df_2d = pd.concat([
    text_df[['headline']], 
    categories.reset_index(drop=True),
    pd.DataFrame(embeds_pc2, columns=['PC1', 'PC2'])
], axis=1)


# In[33]:


# Create 2D scatter plot
scatter_chart = generate_chart(
    df_2d, 'PC1', 'PC2', 'category', 
    '2D Visualization of News Headlines Embeddings'
)

print("2D scatter plot created successfully!")


# In[34]:


# Step 9: Similarity analysis
def find_similar_headlines(query_headline, embeddings_df, top_k=3):
    """Find most similar headlines to a query headline"""
    if query_headline not in embeddings_df['headline'].values:
        print(f"Headline '{query_headline}' not found in dataset")
        return
    
    # Get query embedding
    query_idx = embeddings_df[embeddings_df['headline'] == query_headline].index[0]
    query_embed = np.array(embeddings_df.loc[query_idx, 'headline_embeds']).reshape(1, -1)
    
    # Calculate similarities
    all_embeds = np.array(embeddings_df['headline_embeds'].tolist())
    similarities = cosine_similarity(query_embed, all_embeds)[0]
    
    # Get top similar headlines (excluding the query itself)
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    
    print(f"\nQuery: '{query_headline}'")
    print("Most similar headlines:")
    for i, idx in enumerate(similar_indices, 1):
        similarity_score = similarities[idx]
        similar_headline = embeddings_df.loc[idx, 'headline']
        print(f"{i}. {similar_headline} (similarity: {similarity_score:.3f})")

        # Example similarity search
if len(text_df) > 0:
    sample_headline = text_df['headline'].iloc[0]
    find_similar_headlines(sample_headline, text_df)


# In[35]:


# Step 10: Clustering analysis
def perform_clustering(embeddings_array, n_clusters=3):
    """Perform K-means clustering on embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    return cluster_labels


# In[36]:


# Perform clustering
cluster_labels = perform_clustering(embeds, n_clusters=len(selected_categories))

# Add cluster labels to dataframe
df_clustered = df_2d.copy()
df_clustered['cluster'] = cluster_labels


# In[37]:


print(f"\nClustering Results:")
for cluster_id in range(len(selected_categories)):
    cluster_headlines = df_clustered[df_clustered['cluster'] == cluster_id]['headline'].tolist()
    print(f"\nCluster {cluster_id}:")
    for headline in cluster_headlines[:3]:  # Show first 3
        print(f"  - {headline}")


# In[38]:


# Create clustered visualization
cluster_chart = generate_chart(
    df_clustered, 'PC1', 'PC2', 'cluster', 
    'Clustered News Headlines Embeddings'
)


# In[39]:


print("\n" + "="*50)
print("TUTORIAL COMPLETE!")
print("="*50)
print("\nWhat you've learned:")
print("1. How to load and prepare a text dataset")
print("2. How to generate embeddings using Cohere API")
print("3. How to visualize high-dimensional embeddings")
print("4. How to find similar texts using cosine similarity")
print("5. How to perform clustering on embeddings")
print("\nNext steps:")
print("- Try with your own dataset")
print("- Experiment with different embedding models")
print("- Explore other similarity metrics")
print("- Try different clustering algorithms")


# In[ ]:




