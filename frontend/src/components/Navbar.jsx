import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './custom/Navbar.css'; // Make sure this includes your updated CSS

const Navbar = () => {
  const [click, setClick] = useState(false);
  const [navbarScrolled, setNavbarScrolled] = useState(false);

  const handleClick = () => setClick(!click);
  const closeMobileMenu = () => setClick(false);

  // Function to scroll to top smoothly
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Lock scroll when mobile menu is open
  useEffect(() => {
    if (click) {
      document.body.classList.add('no-scroll');
    } else {
      document.body.classList.remove('no-scroll');
    }
    return () => document.body.classList.remove('no-scroll');
  }, [click]);

  // Scroll-aware navbar shadow effect
  useEffect(() => {
    const handleScroll = () => {
      setNavbarScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`navbar ${navbarScrolled ? 'navbar-scrolled' : ''}`} role="navigation" aria-label="Main Navigation">
      <div className="navbar-container">
        <Link 
          to="/" 
          className="navbar-logo" 
          onClick={() => { // Modified onClick for logo
            closeMobileMenu();
            scrollToTop(); // Scroll to top when logo is clicked
          }} 
          aria-label="Home"
        >
          🔮 StockSense
        </Link>

        <div
          className="menu-icon"
          onClick={handleClick}
          aria-label="Toggle navigation menu"
          aria-expanded={click}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') handleClick();
          }}
        >
          <i className={click ? 'fas fa-times' : 'fas fa-bars'} />
        </div>

        <ul className={click ? 'nav-menu active' : 'nav-menu'}>
          <li className="nav-item">
            <Link 
              to="/" 
              className="nav-links" 
              onClick={() => { // Modified onClick for Home link
                closeMobileMenu();
                scrollToTop(); // Scroll to top when Home link is clicked
              }}
            >
              Home
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/#about-section" className="nav-links" onClick={closeMobileMenu}>
              About
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/stocks" className="nav-links" onClick={closeMobileMenu}>
              Stocks
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/predict" className="nav-links" onClick={closeMobileMenu}>
              Predict
            </Link>
          </li>
          
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;