To develop the functionality for users to leave ratings and reviews on traders, we will create a RESTful API using Flask, a popular Python web framework. We'll also use SQLAlchemy for database interactions. The solution will include endpoints for submitting and retrieving ratings and reviews, and the data will be stored in a relational database.

Here's the complete code:

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traders.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model for Trader
class Trader(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ratings = db.relationship('Rating', backref='trader', lazy=True)

# Database model for Rating
class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trader_id = db.Column(db.Integer, db.ForeignKey('trader.id'), nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    review = db.Column(db.String(500), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'trader_id': self.trader_id,
            'user_id': self.user_id,
            'score': self.score,
            'review': self.review
        }

# Initialize the database
@app.before_first_request
def create_tables():
    db.create_all()

# API endpoint to submit a rating
@app.route('/traders/<int:trader_id>/ratings', methods=['POST'])
def submit_rating(trader_id):
    data = request.get_json()
    user_id = data.get('user_id')
    score = data.get('score')
    review = data.get('review', '')

    if not (1 <= score <= 5):
        return jsonify({'error': 'Score must be between 1 and 5'}), 400

    rating = Rating(trader_id=trader_id, user_id=user_id, score=score, review=review)
    db.session.add(rating)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Failed to submit rating'}), 500

    return jsonify({'message': 'Rating submitted successfully'}), 201

# API endpoint to retrieve ratings for a trader
@app.route('/traders/<int:trader_id>/ratings', methods=['GET'])
def get_ratings(trader_id):
    ratings = Rating.query.filter_by(trader_id=trader_id).all()
    return jsonify([rating.to_dict() for rating in ratings]), 200

if __name__ == '__main__':
    app.run(debug=True)
```

### Explanation:

1. **Database Models**: We have two models, `Trader` and `Rating`. The `Trader` model represents traders, and the `Rating` model represents user ratings and reviews for traders. The `Rating` model includes a foreign key to the `Trader` model.

2. **API Endpoints**:
   - **Submit Rating**: The endpoint `/traders/<int:trader_id>/ratings` accepts POST requests to submit a rating. It expects a JSON payload with `user_id`, `score`, and an optional `review`. The score must be between 1 and 5.
   - **Retrieve Ratings**: The endpoint `/traders/<int:trader_id>/ratings` accepts GET requests to retrieve all ratings for a specific trader.

3. **Database Initialization**: The `create_tables` function initializes the database tables before the first request.

4. **Error Handling**: Basic error handling is implemented for score validation and database integrity errors.

This code is modular, testable, and follows best practices in software design. You can further extend it with authentication, more detailed error handling, and additional features as needed.