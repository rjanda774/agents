To create a leaderboard that displays the top-performing traders based on user engagement and trading success metrics, we need to develop both backend and frontend components. The backend will handle data processing and calculations, while the frontend will display the leaderboard using React. Below is a modular and testable solution:

### Backend (Python with Flask)

1. **Setup Flask Application**: Create a Flask application to handle API requests.

2. **Data Model**: Define a data model for traders, including fields for user engagement and trading success metrics.

3. **Leaderboard Logic**: Implement logic to calculate the top-performing traders.

4. **API Endpoint**: Create an API endpoint to fetch the leaderboard data.

```python
from flask import Flask, jsonify
from operator import itemgetter

app = Flask(__name__)

# Sample data for traders
traders = [
    {'id': 1, 'name': 'Trader A', 'engagement': 80, 'success': 90},
    {'id': 2, 'name': 'Trader B', 'engagement': 85, 'success': 85},
    {'id': 3, 'name': 'Trader C', 'engagement': 70, 'success': 95},
    # Add more traders as needed
]

def calculate_leaderboard(traders):
    # Sort traders based on engagement and success metrics
    sorted_traders = sorted(traders, key=lambda x: (x['engagement'], x['success']), reverse=True)
    return sorted_traders

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    leaderboard = calculate_leaderboard(traders)
    return jsonify(leaderboard)

if __name__ == '__main__':
    app.run(debug=True)
```

### Frontend (React)

1. **Setup React Application**: Use `create-react-app` to set up a new React application.

2. **Fetch Leaderboard Data**: Use `fetch` or `axios` to call the API endpoint and retrieve leaderboard data.

3. **Display Leaderboard**: Create a component to display the leaderboard.

```jsx
import React, { useEffect, useState } from 'react';

function Leaderboard() {
  const [traders, setTraders] = useState([]);

  useEffect(() => {
    fetch('/api/leaderboard')
      .then(response => response.json())
      .then(data => setTraders(data))
      .catch(error => console.error('Error fetching leaderboard:', error));
  }, []);

  return (
    <div>
      <h1>Top Performing Traders</h1>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Engagement</th>
            <th>Success</th>
          </tr>
        </thead>
        <tbody>
          {traders.map(trader => (
            <tr key={trader.id}>
              <td>{trader.name}</td>
              <td>{trader.engagement}</td>
              <td>{trader.success}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Leaderboard;
```

### Best Practices

- **Modular Code**: The backend and frontend are separated into different modules, making the codebase easier to maintain and test.
- **Testability**: The leaderboard calculation logic is isolated in a function, which can be easily tested with unit tests.
- **Scalability**: The solution can be extended to include more complex metrics and additional features as needed.

This solution provides a simple and effective way to display a leaderboard of top-performing traders, ensuring that the code is modular, testable, and maintainable.