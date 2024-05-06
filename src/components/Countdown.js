import React, { useState, useEffect } from 'react';

const Countdown = () => {
  const [seconds, setSeconds] = useState(30);

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prevSeconds => {
        if (prevSeconds === 0) {
          clearInterval(interval); // Stop the interval when seconds reach 0
          return prevSeconds;
        }
        return prevSeconds - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h1>Countdown</h1>
      <div className="countdown">{seconds}</div>
    </div>
  );
};

export default Countdown;
