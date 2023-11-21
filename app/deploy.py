from flask import Flask,render_template,request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize




app=Flask(__name__)

df=pd.read_csv('data.csv')

def rec(course):
    
    if course not in df['Course name'].values:
        return [{'course_name': 'Course not found', 'url': ''}]
    
    recommendations=[]
    course_domain = df[df['Course name'] == course]['domain'].iloc[0]
    course_df=df[df['domain']==course_domain].reset_index()
    course_index=course_df[course_df['Course name'] == course].index[0]
    
    
    
    cv=CountVectorizer(max_features=10000, stop_words='english')
    vectors=cv.fit_transform(course_df['stemmed']).toarray()
    vectors_normalized = normalize(vectors)
    
    similarity=cosine_similarity([vectors_normalized[course_index]],vectors_normalized)
    distances=similarity[0]
    
    course_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:16]
    for i in course_list:
        course_name=course_df['Course name'][i[0]]
        url=course_df['URL'][i[0]]
        recommendations.append({'course_name':course_name, 'url':url})
    return recommendations
    
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search',methods=['POST'])
def search():
    search_term=request.form['search']
    recommendations=rec(search_term)
    return render_template('results.html',search_term=search_term,recommendations=recommendations)
if __name__ == '__main__':
    app.run(debug=True)
    
    


    