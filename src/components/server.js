const express = require('express');
const fs = require('fs');
const app = express();
const port = 3500;

app.use(express.json());

app.post('/api/login', (req, res) => {
  const userData = JSON.stringify(req.body);

  fs.writeFile('./src/users/user.json', userData, (err) => {
    if (err) {
      console.error('Error writing file:', err);
      res.status(500).send('Error writing file');
    } else {
      console.log('User data saved successfully');
      res.status(200).send('User data saved successfully');
    }
  });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});