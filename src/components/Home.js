import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
    return (
        <div className="container mt-5">
            <h1 className="text-center">Home Page</h1>
            <div className="card mx-auto" style={{ maxWidth: '300px' }}>
                <div className="card-body">
                    <h2 className="card-title text-center">Options:</h2>
                    <ul className="list-group list-group-flush">
                        <li className="list-group-item">
                            <Link to="/record-session" className="btn btn-primary btn-block">Record Session</Link>
                        </li>
                    </ul>
                </div>
                
            </div>
        </div>
    );
};

export default HomePage;
