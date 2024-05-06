import React, { useState } from 'react';
import axios from 'axios';

const Login = () => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');

  const handleLogin = async () => {
    try {
      const response = await axios.post('http://localhost:3001/api/login', {
        name,
        age,
        gender
      });
      console.log('User data saved successfully');
    } catch (error) {
      console.error('Error saving user data:', error);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">Login Page</h1>
      <form>
        <div className="form-group">
          <label>Name:</label>
          <input type="text" className="form-control" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="form-group">
          <label>Age:</label>
          <input type="number" className="form-control" value={age} onChange={(e) => setAge(e.target.value)} />
        </div>
        <div className="form-group">
          <label>Gender:</label>
          <select className="form-control" value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="">Select Gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
        <button type="button" className="btn btn-primary" onClick={handleLogin}>Login</button>
      </form>
    </div>
  );
};

export default Login;