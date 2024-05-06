import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar.js';
// import Home from './components/Home.js';
// import ImageView from './components/ImageView.js';
// import ImageUpload from './components/ImageUpload.js';
import RealTimeData from './components/RealTimeData';
import 'bootstrap/dist/css/bootstrap.min.css';
import HomePage from "./components/Home";
import Login from "./components/Login";
import DisplayResults from "./components/DisplayResults.js"

function App() {
  return (
    <BrowserRouter>
      <Navbar />
        <Routes>
        <Route index element={<HomePage />} />
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/record-session" element={<RealTimeData />} />
        <Route path="/display" element={<DisplayResults />} />
      </Routes>
    </BrowserRouter>

  );
}

export default App;
// import logo from './logo.svg';
// import './App.css';
// import { RealTimeChart } from './components/RealTimeChart';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//           <RealTimeChart />
//       </header>
//     </div>
//   );
// }

// export default App;
