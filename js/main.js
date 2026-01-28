/**
 * SynthVerse Website - Main JavaScript
 */

// ============================================
// Navbar Scroll Effect
// ============================================
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ============================================
// Smooth Scrolling for Navigation Links
// ============================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#' && href.length > 1) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navbarHeight = document.getElementById('navbar').offsetHeight;
                const targetPosition = target.offsetTop - navbarHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        }
    });
});

// ============================================
// Copy BibTeX Citation
// ============================================
function copyBibTeX() {
    const citationText = `@inproceedings{synthverse2025,
  title={SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking},
  author={Zhao, Weiguang and Xu, Haoran and Miao, Xingyu and Zhao, Qin and Zhang, Rui and Huang, Kaizhu and Gao, Ning and Cao, Peizhou and Sun, Mingze and Yu, Mulin and Lu, Tao and Xu, Linning and Dong, Junting and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}`;

    const copyBtn = document.querySelector('.copy-btn');
    const originalHTML = copyBtn.innerHTML;

    function showSuccess() {
        copyBtn.innerHTML = `
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                <path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/>
            </svg>
            Copied!
        `;
        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
        }, 2000);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(citationText).then(showSuccess).catch(() => {
            fallbackCopy(citationText, showSuccess);
        });
    } else {
        fallbackCopy(citationText, showSuccess);
    }
}

function fallbackCopy(text, onSuccess) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        onSuccess();
    } catch (err) {
        alert('Failed to copy citation. Please copy manually.');
    }
    document.body.removeChild(textarea);
}

// ============================================
// Fade-in Animation on Scroll
// ============================================
function fadeInOnScroll() {
    const elements = document.querySelectorAll('.feature-card, .stat-card, .scene-type-card, .finding-card, .download-card, .video-container, .pipeline-step, .table-wrapper');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '0';
                entry.target.style.transform = 'translateY(20px)';
                entry.target.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 100);

                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    elements.forEach(element => {
        observer.observe(element);
    });
}

// ============================================
// Active Navigation Link Highlighting
// ============================================
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const navbarHeight = document.getElementById('navbar').offsetHeight;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.scrollY >= (sectionTop - navbarHeight - 100)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ============================================
// Lazy Loading Images
// ============================================
function lazyLoadImages() {
    const images = document.querySelectorAll('img[loading="lazy"]');

    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.add('loaded');
                    observer.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    }
}

// ============================================
// Mobile Menu Toggle
// ============================================
function toggleMobileMenu() {
    const navLinks = document.getElementById('nav-links');
    const menuBtn = document.querySelector('.mobile-menu-btn');
    navLinks.classList.toggle('active');
    menuBtn.classList.toggle('active');
}

function initMobileMenu() {
    // Close mobile menu when a nav link is clicked
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            const navLinks = document.getElementById('nav-links');
            const menuBtn = document.querySelector('.mobile-menu-btn');
            if (navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
            }
        });
    });
}

// ============================================
// Initialize All Functions
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    fadeInOnScroll();
    updateActiveNavLink();
    lazyLoadImages();
    initMobileMenu();
    initVizIframe();

    console.log('🚀 SynthVerse website loaded successfully!');
});

// ============================================
// Viz Iframe Loading Handler
// ============================================
function initVizIframe() {
    const vizFrame = document.getElementById('vizFrame');
    const vizLoader = document.getElementById('vizLoader');

    if (vizFrame && vizLoader) {
        // Detect file:// protocol — 3D viz requires HTTP server
        if (window.location.protocol === 'file:') {
            vizLoader.innerHTML =
                '<p style="text-align:center;max-width:500px;padding:0 1rem;color:#6b7280;line-height:1.7;">' +
                '<strong style="color:#1f2937;font-size:1.1rem;">3D Visualizer requires HTTP server</strong><br><br>' +
                'The interactive 3D viewer cannot load data via <code>file://</code> protocol.<br>' +
                'Please start a local server in the project root:<br><br>' +
                '<code style="background:#f3f4f6;padding:4px 10px;border-radius:4px;font-size:0.9rem;">python -m http.server 8000</code><br><br>' +
                'Then open <code style="background:#f3f4f6;padding:4px 10px;border-radius:4px;font-size:0.9rem;">http://localhost:8000</code></p>';
            return;
        }

        vizFrame.addEventListener('load', function() {
            vizLoader.classList.add('hidden');
        });

        // Fallback: hide loader after 10 seconds even if load event doesn't fire
        setTimeout(() => {
            if (vizLoader) {
                vizLoader.classList.add('hidden');
            }
        }, 10000);
    }
}

// ============================================
// Performance Optimization
// ============================================
// Debounce function for scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debounce to scroll-heavy functions
const debouncedScroll = debounce(() => {
    // Any heavy scroll operations can go here
}, 100);

window.addEventListener('scroll', debouncedScroll);

// ============================================
// Scene Thumbnail Selection
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const thumbnailWrappers = document.querySelectorAll('.scene-thumbnail-wrapper');
    const thumbnails = document.querySelectorAll('.scene-thumbnail');
    const vizFrame = document.getElementById('vizFrame');
    
    if (thumbnailWrappers.length > 0 && vizFrame) {
        // Handle click on wrapper
        thumbnailWrappers.forEach(wrapper => {
            wrapper.addEventListener('click', function() {
                const scene = this.dataset.scene;
                if (scene) {
                    // Update active state on thumbnails
                    thumbnails.forEach(t => t.classList.remove('active'));
                    const thumb = this.querySelector('.scene-thumbnail');
                    if (thumb) thumb.classList.add('active');
                    
                    // Update iframe src to load new scene
                    vizFrame.src = `demos/viz.html?scene=${scene}`;
                }
            });
        });
    }
});
