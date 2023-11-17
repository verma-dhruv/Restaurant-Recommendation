recommendations = recommend(user_input)
    return jsonify(recommendations.to_dict(orient='records'))