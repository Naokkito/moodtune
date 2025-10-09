// Initialize Lucide icons
lucide.createIcons();

// App State
const state = {
    recommendations: [],
    favorites: JSON.parse(localStorage.getItem('moodtune_favorites')) || [],
    currentMood: '',
    isLoading: false,
    currentTrackIndex: -1,
    isPlaying: false,
    audioPlayer: null
};

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const elements = {
    moodInput: document.getElementById('mood-input'),
    searchBtn: document.getElementById('search-btn'),
    resultsSection: document.getElementById('results-section'),
    tracksContainer: document.getElementById('tracks-container'),
    currentMood: document.getElementById('current-mood'),
    loading: document.getElementById('loading'),
    exampleChips: document.getElementById('example-chips'),
    favoriteCount: document.getElementById('favorite-count'),
    resultsCount: document.getElementById('results-count')
};

// Mood Examples
const moodExamples = [
    "focus study music",
    "chill relaxing vibes", 
    "energy workout songs",
    "happy summer party",
    "romantic evening dinner",
    "sad emotional moments",
    "party dance music",
    "coding focus session"
];

// Initialize app
function init() {
    generateExamples();
    updateFavoriteCount();
    testBackendConnection();
}

// Test backend connection
async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend connected successfully');
        }
    } catch (error) {
        console.warn('❌ Backend not reachable:', error);
    }
}

// Generate example chips
function generateExamples() {
    elements.exampleChips.innerHTML = '';
    moodExamples.forEach(example => {
        const chip = document.createElement('button');
        chip.className = 'example-chip';
        chip.innerHTML = example;
        chip.onclick = () => useExample(example);
        elements.exampleChips.appendChild(chip);
    });
}

// Use example
function useExample(text) {
    elements.moodInput.value = text;
    getRecommendations();
}

// Handle Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        getRecommendations();
    }
}

// Get recommendations
async function getRecommendations() {
    const moodText = elements.moodInput.value.trim();
    
    if (!moodText) {
        alert('Please describe your mood');
        return;
    }

    setLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend-with-songs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: moodText,
                limit: 8
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.personalized_recommendations && data.personalized_recommendations.length > 0) {
            displayResults(data.personalized_recommendations, moodText, data.top_5_songs_for_this_vibe);
        } else {
            throw new Error('No recommendations received');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Failed to get recommendations: ${error.message}\n\nMake sure the backend is running on ${API_BASE_URL}`);
    } finally {
        setLoading(false);
    }
}

// Set loading state
function setLoading(loading) {
    state.isLoading = loading;
    elements.searchBtn.disabled = loading;
    
    if (loading) {
        elements.loading.style.display = 'block';
        elements.searchBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i><span>Generating...</span>';
        elements.resultsSection.style.display = 'none';
    } else {
        elements.loading.style.display = 'none';
        elements.searchBtn.innerHTML = '<i data-lucide="zap"></i><span>Generate</span>';
    }
    lucide.createIcons();
}

// Display results
function displayResults(recommendations, moodText, topSongs = []) {
    state.recommendations = recommendations;
    state.currentMood = moodText;
    
    elements.currentMood.textContent = `"${moodText}"`;
    elements.resultsCount.textContent = `${recommendations.length} tracks matching your vibe`;
    
    // Clear previous results
    elements.tracksContainer.innerHTML = '';
    
    // Add top songs section if available
    if (topSongs && topSongs.length > 0) {
        const topSongsSection = createTopSongsSection(topSongs);
        elements.tracksContainer.appendChild(topSongsSection);
    }
    
    // Add personalized recommendations
    recommendations.forEach((track, index) => {
        const trackElement = createTrackElement(track, index);
        elements.tracksContainer.appendChild(trackElement);
    });
    
    // Show results
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Create top songs section
function createTopSongsSection(topSongs) {
    const section = document.createElement('div');
    section.className = 'top-songs-section';
    section.innerHTML = `
        <div class="section-header">
            <h4>🎯 Top Songs for This Vibe</h4>
            <p>Most popular tracks matching your mood</p>
        </div>
        <div class="top-songs-grid">
            ${topSongs.map(song => `
                <div class="top-song-card">
                    <div class="song-rank">${song.rank}</div>
                    <div class="song-info">
                        <div class="song-title">${song.song_name}</div>
                        <div class="song-artist">${song.artist_name}</div>
                    </div>
                    <div class="song-match">${song.match_percentage}%</div>
                </div>
            `).join('')}
        </div>
    `;
    
    // Add CSS for top songs section
    if (!document.querySelector('#top-songs-styles')) {
        const style = document.createElement('style');
        style.id = 'top-songs-styles';
        style.textContent = `
            .top-songs-section {
                margin-bottom: 2rem;
                padding: 1.5rem;
                background: var(--surface-light);
                border-radius: 16px;
                border: 1px solid var(--border);
            }
            .section-header {
                text-align: center;
                margin-bottom: 1.5rem;
            }
            .section-header h4 {
                font-size: 1.25rem;
                margin-bottom: 0.5rem;
                color: var(--text-primary);
            }
            .section-header p {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
            .top-songs-grid {
                display: grid;
                gap: 0.75rem;
            }
            .top-song-card {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem;
                background: var(--surface);
                border-radius: 12px;
                border: 1px solid var(--border-light);
            }
            .song-rank {
                background: var(--gradient-primary);
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.9rem;
            }
            .song-info {
                flex: 1;
            }
            .song-title {
                font-weight: 600;
                font-size: 0.95rem;
            }
            .song-artist {
                color: var(--text-secondary);
                font-size: 0.85rem;
            }
            .song-match {
                background: rgba(204, 221, 203, 0.3);
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-weight: 700;
                font-size: 0.85rem;
            }
        `;
        document.head.appendChild(style);
    }
    
    return section;
}

// Create track element
function createTrackElement(track, index) {
    const isFavorite = state.favorites.some(fav => fav.track_id === track.track_id);
    
    const trackElement = document.createElement('div');
    trackElement.className = 'track-card';
    trackElement.innerHTML = `
        <div class="track-number">${index + 1}</div>
        <div class="track-info">
            <div class="track-title">${track.song_name || track.track_name}</div>
            <div class="track-artist">${track.artist_name || (Array.isArray(track.artists) ? track.artists.join(', ') : track.artists)}</div>
            ${track.why_it_matches ? `<div class="track-reason">${track.why_it_matches}</div>` : ''}
        </div>
        <div class="track-meta">
            <div class="similarity-score">
                ${track.match_percentage || Math.round((track.similarity_score || 0) * 100)}% match
            </div>
            <div class="track-actions">
                <button class="btn-action favorite ${isFavorite ? 'active' : ''}" onclick="toggleFavorite(${JSON.stringify(track).replace(/"/g, '&quot;')})">
                    <i data-lucide="heart" ${isFavorite ? 'fill="currentColor"' : ''}></i>
                </button>
                ${track.preview_url ? `
                    <button class="btn-action play" onclick="playPreview('${track.preview_url}', ${index})">
                        <i data-lucide="play"></i>
                    </button>
                ` : ''}
            </div>
        </div>
    `;
    
    return trackElement;
}

// Toggle favorite
function toggleFavorite(track) {
    const index = state.favorites.findIndex(fav => fav.track_id === track.track_id);
    
    if (index > -1) {
        state.favorites.splice(index, 1);
    } else {
        state.favorites.push(track);
    }
    
    localStorage.setItem('moodtune_favorites', JSON.stringify(state.favorites));
    updateFavoriteCount();
    displayResults(state.recommendations, state.currentMood);
}

// Update favorite count
function updateFavoriteCount() {
    elements.favoriteCount.textContent = state.favorites.length;
}

// Play preview (placeholder - you can implement actual audio playback)
function playPreview(url, index) {
    alert(`Would play preview: ${url}\n\nTrack ${index + 1}`);
    // Implementation for actual audio playback:
    // if (state.audioPlayer) {
    //     state.audioPlayer.pause();
    // }
    // state.audioPlayer = new Audio(url);
    // state.audioPlayer.play();
    // state.isPlaying = true;
    // state.currentTrackIndex = index;
}

// Back to search
function backToSearch() {
    elements.resultsSection.style.display = 'none';
    elements.moodInput.focus();
}

// Clear results
function clearResults() {
    elements.resultsSection.style.display = 'none';
    elements.moodInput.value = '';
    elements.moodInput.focus();
}

// Export playlist
function exportPlaylist() {
    if (state.recommendations.length === 0) {
        alert('No tracks to export');
        return;
    }
    
    const playlist = {
        name: `MoodTune - ${state.currentMood}`,
        timestamp: new Date().toISOString(),
        tracks: state.recommendations
    };
    
    const dataStr = JSON.stringify(playlist, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `moodtune-${state.currentMood.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.json`;
    link.click();
    
    alert('Playlist exported!');
}

// Toggle theme
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // Update icon
    const themeButton = document.querySelector('.btn-icon [data-lucide]');
    if (themeButton) {
        const newIcon = newTheme === 'dark' ? 'sun' : 'moon';
        themeButton.parentElement.innerHTML = `<i data-lucide="${newIcon}"></i>`;
        lucide.createIcons();
    }
}

// Show about
function showAbout() {
    alert('MoodTune - AI Music Curation\n\nDescribe any mood, scene, or activity to get personalized music recommendations with real song names and artists.');
}

// Show privacy
function showPrivacy() {
    alert('Privacy: Your data stays on your device. No information is sent to external servers except for music recommendations.');
}

// Show terms
function showTerms() {
    alert('Terms: Use responsibly. Music recommendations are for personal use.');
}

// Show favorites
function showFavorites() {
    if (state.favorites.length === 0) {
        alert('No favorites yet! Add some tracks to your favorites.');
        return;
    }
    
    let favoritesText = 'Your Favorite Tracks:\n\n';
    state.favorites.forEach((track, index) => {
        favoritesText += `${index + 1}. ${track.song_name || track.track_name} - ${track.artist_name || (Array.isArray(track.artists) ? track.artists.join(', ') : track.artists)}\n`;
    });
    
    alert(favoritesText);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);