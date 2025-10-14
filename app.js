
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

// Mood Word Banks for Random Generation
const moodWords = {
    scenes: [
        'coffee shop', 'rainy day', 'sunny beach', 'forest walk', 'mountain top',
        'city lights', 'space station', 'cozy cabin', 'desert road', 'lake house',
        'winter night', 'spring garden', 'summer party', 'autumn park', 'tropical island',
        'bookstore', 'art gallery', 'train journey', 'campfire', 'starry night'
    ],
    activities: [
        'morning', 'evening', 'night', 'afternoon', 'study session', 'workout',
        'road trip', 'yoga', 'meditation', 'coding', 'writing', 'painting',
        'cooking', 'cleaning', 'showering', 'commuting', 'gaming', 'dancing',
        'reading', 'dreaming', 'exploring', 'creating', 'reflecting', 'celebrating'
    ],
    emotions: [
        'happy', 'sad', 'energetic', 'calm', 'focused', 'relaxed', 'nostalgic',
        'hopeful', 'romantic', 'melancholic', 'epic', 'mysterious', 'dreamy',
        'intense', 'peaceful', 'joyful', 'thoughtful', 'adventurous', 'cozy',
        'powerful', 'serene', 'lonely', 'triumphant', 'yearning'
    ],
    genres: [
        'lofi', 'jazz', 'electronic', 'classical', 'rock', 'pop', 'ambient',
        'synthwave', 'folk', 'indie', 'r&b', 'hip hop', 'reggae', 'blues',
        'orchestral', 'chillhop', 'downtempo', 'house', 'techno', 'acoustic'
    ],
    intensities: [
        'soft', 'gentle', 'moderate', 'intense', 'powerful', 'explosive',
        'mellow', 'subtle', 'strong', 'dynamic', 'building', 'crescendo'
    ],
    times: [
        'sunrise', 'sunset', 'midnight', 'dawn', 'dusk', 'golden hour',
        'blue hour', 'afternoon', 'early morning', 'late night'
    ]
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

// Initialize app
function init() {
    generateRandomExamples();
    updateFavoriteCount();
    testBackendConnection();

    // Add click listener to refresh examples
    document.querySelector('.examples').addEventListener('click', function(e) {
        if (e.target.classList.contains('refresh-examples')) {
            generateRandomExamples();
        }
    });
}

// Test backend connection
async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connected successfully:', data);
        } else {
            console.warn('❌ Backend responded with error:', response.status);
        }
    } catch (error) {
        console.warn('❌ Backend not reachable:', error);
        // Show user-friendly message
        setTimeout(() => {
            if (confirm('Backend server is not running. Would you like to see instructions on how to start it?')) {
                showBackendInstructions();
            }
        }, 1000);
    }
}

function showBackendInstructions() {
    alert('To start the backend server:\n\n1. Open terminal/command prompt\n2. Navigate to the project folder\n3. Run: python moodtune_backend.py\n4. Wait for "Starting FastAPI server..." message\n5. Then refresh this page');
}

// Generate Random Examples
function generateRandomExamples() {
    const numberOfExamples = 4;
    elements.exampleChips.innerHTML = '';

    for (let i = 0; i < numberOfExamples; i++) {
        const example = generateRandomMood();
        const chip = createExampleChip(example, i);
        elements.exampleChips.appendChild(chip);
    }

    // Add refresh button
    const refreshChip = document.createElement('button');
    refreshChip.className = 'example-chip refresh-examples';
    refreshChip.innerHTML = '<i data-lucide="refresh-cw"></i> New Ideas';
    refreshChip.title = 'Generate new random examples';
    elements.exampleChips.appendChild(refreshChip);

    lucide.createIcons();
}

function generateRandomMood() {
    const templates = [
        // Scene + Activity
        () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.activities)}`,
        // Emotion + Activity
        () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.activities)}`,
        // Scene + Time
        () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`,
        // Emotion + Scene
        () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.scenes)}`,
        // Intensity + Emotion + Genre
        () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.emotions)} ${randomWord(moodWords.genres)}`,
        // Time + Activity + Scene
        () => `${randomWord(moodWords.times)} ${randomWord(moodWords.activities)} ${randomWord(moodWords.scenes)}`,
        // Genre + Scene + Emotion
        () => `${randomWord(moodWords.genres)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.emotions)}`,
        // Multiple emotions
        () => `${randomWord(moodWords.emotions)} and ${randomWord(moodWords.emotions)}`,
        // Complex scene description
        () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`
    ];

    const template = randomWord(templates);
    return template();
}

function randomWord(wordArray) {
    return wordArray[Math.floor(Math.random() * wordArray.length)];
}

function createExampleChip(example, index) {
    const chip = document.createElement('button');
    chip.className = 'example-chip';
    chip.innerHTML = `<span class="example-emoji">${getRandomEmoji()}</span> ${example}`;

    chip.addEventListener('click', () => {
        useExample(example);
        // Add a subtle animation when clicked
        chip.style.transform = 'scale(0.95)';
        setTimeout(() => chip.style.transform = 'scale(1)', 150);
    });

    // Staggered animation
    chip.style.animationDelay = `${index * 0.1}s`;

    return chip;
}

function getRandomEmoji() {
    const emojis = ['🎵', '🎶', '🎧', '🎸', '🎹', '🥁', '🎷', '🎺', '🪕', '🎻', '✨', '🌟', '💫', '🔥', '💧', '🌊', '🍃', '🌙', '⭐', '⚡', '❤️', '🎉', '🌈', '🎨'];
    return randomWord(emojis);
}

// Use example
function useExample(text) {
    elements.moodInput.value = text;
    elements.moodInput.focus();
    // Auto-generate after a short delay to show the input change
    setTimeout(() => getRecommendations(), 300);
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
        console.log('Sending request to backend...');
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
        console.log('Received data from backend:', data);

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
    return section;
}

// Create track element - FIXED to handle artist_name properly
function createTrackElement(track, index) {
    // Use artist_name from backend response
    const artistName = track.artist_name || 'Unknown Artist';
    const songName = track.song_name || 'Unknown Track';
    const trackId = track.track_id || `track-${index}-${Date.now()}`;

    const isFavorite = state.favorites.some(fav => fav.track_id === trackId);

    const trackElement = document.createElement('div');
    trackElement.className = 'track-card';
    trackElement.innerHTML = `
        <div class="track-number">${index + 1}</div>
        <div class="track-info">
            <div class="track-title">${songName}</div>
            <div class="track-artist">${artistName}</div>
            ${track.why_it_matches ? `<div class="track-reason">${track.why_it_matches}</div>` : ''}
        </div>
        <div class="track-meta">
            <div class="similarity-score">
                ${track.match_percentage || Math.round((track.similarity_score || 0.8) * 100)}% match
            </div>
            <div class="track-actions">
                <button class="btn-action favorite ${isFavorite ? 'active' : ''}" onclick="toggleFavorite(${JSON.stringify({...track, track_id: trackId, artist_name: artistName, song_name: songName}).replace(/"/g, '&quot;')})">
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
    const trackId = track.track_id;
    const index = state.favorites.findIndex(fav => fav.track_id === trackId);

    if (index > -1) {
        state.favorites.splice(index, 1);
    } else {
        state.favorites.push(track);
    }

    localStorage.setItem('moodtune_favorites', JSON.stringify(state.favorites));
    updateFavoriteCount();

    // Re-render the current results to update heart icons
    if (state.recommendations.length > 0) {
        displayResults(state.recommendations, state.currentMood);
    }
}

// Update favorite count
function updateFavoriteCount() {
    elements.favoriteCount.textContent = state.favorites.length;
}

// Play preview
function playPreview(url, index) {
    alert(`Would play preview: ${url}\n\nTrack ${index + 1}`);
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
        const songName = track.song_name || track.name || 'Unknown Track';
        const artistName = track.artist_name || track.artist || 'Unknown Artist';
        favoritesText += `${index + 1}. ${songName} - ${artistName}\n`;
    });

    alert(favoritesText);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);
</script><script>
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

// Mood Word Banks for Random Generation
const moodWords = {
    scenes: [
        'coffee shop', 'rainy day', 'sunny beach', 'forest walk', 'mountain top',
        'city lights', 'space station', 'cozy cabin', 'desert road', 'lake house',
        'winter night', 'spring garden', 'summer party', 'autumn park', 'tropical island',
        'bookstore', 'art gallery', 'train journey', 'campfire', 'starry night'
    ],
    activities: [
        'morning', 'evening', 'night', 'afternoon', 'study session', 'workout',
        'road trip', 'yoga', 'meditation', 'coding', 'writing', 'painting',
        'cooking', 'cleaning', 'showering', 'commuting', 'gaming', 'dancing',
        'reading', 'dreaming', 'exploring', 'creating', 'reflecting', 'celebrating'
    ],
    emotions: [
        'happy', 'sad', 'energetic', 'calm', 'focused', 'relaxed', 'nostalgic',
        'hopeful', 'romantic', 'melancholic', 'epic', 'mysterious', 'dreamy',
        'intense', 'peaceful', 'joyful', 'thoughtful', 'adventurous', 'cozy',
        'powerful', 'serene', 'lonely', 'triumphant', 'yearning'
    ],
    genres: [
        'lofi', 'jazz', 'electronic', 'classical', 'rock', 'pop', 'ambient',
        'synthwave', 'folk', 'indie', 'r&b', 'hip hop', 'reggae', 'blues',
        'orchestral', 'chillhop', 'downtempo', 'house', 'techno', 'acoustic'
    ],
    intensities: [
        'soft', 'gentle', 'moderate', 'intense', 'powerful', 'explosive',
        'mellow', 'subtle', 'strong', 'dynamic', 'building', 'crescendo'
    ],
    times: [
        'sunrise', 'sunset', 'midnight', 'dawn', 'dusk', 'golden hour',
        'blue hour', 'afternoon', 'early morning', 'late night'
    ]
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

// Initialize app
function init() {
    generateRandomExamples();
    updateFavoriteCount();
    testBackendConnection();

    // Add click listener to refresh examples
    document.querySelector('.examples').addEventListener('click', function(e) {
        if (e.target.classList.contains('refresh-examples')) {
            generateRandomExamples();
        }
    });
}

// Test backend connection
async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connected successfully:', data);
        } else {
            console.warn('❌ Backend responded with error:', response.status);
        }
    } catch (error) {
        console.warn('❌ Backend not reachable:', error);
        // Show user-friendly message
        setTimeout(() => {
            if (confirm('Backend server is not running. Would you like to see instructions on how to start it?')) {
                showBackendInstructions();
            }
        }, 1000);
    }
}

function showBackendInstructions() {
    alert('To start the backend server:\n\n1. Open terminal/command prompt\n2. Navigate to the project folder\n3. Run: python moodtune_backend.py\n4. Wait for "Starting FastAPI server..." message\n5. Then refresh this page');
}

// Generate Random Examples
function generateRandomExamples() {
    const numberOfExamples = 4;
    elements.exampleChips.innerHTML = '';

    for (let i = 0; i < numberOfExamples; i++) {
        const example = generateRandomMood();
        const chip = createExampleChip(example, i);
        elements.exampleChips.appendChild(chip);
    }

    // Add refresh button
    const refreshChip = document.createElement('button');
    refreshChip.className = 'example-chip refresh-examples';
    refreshChip.innerHTML = '<i data-lucide="refresh-cw"></i> New Ideas';
    refreshChip.title = 'Generate new random examples';
    elements.exampleChips.appendChild(refreshChip);

    lucide.createIcons();
}

function generateRandomMood() {
    const templates = [
        // Scene + Activity
        () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.activities)}`,
        // Emotion + Activity
        () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.activities)}`,
        // Scene + Time
        () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`,
        // Emotion + Scene
        () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.scenes)}`,
        // Intensity + Emotion + Genre
        () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.emotions)} ${randomWord(moodWords.genres)}`,
        // Time + Activity + Scene
        () => `${randomWord(moodWords.times)} ${randomWord(moodWords.activities)} ${randomWord(moodWords.scenes)}`,
        // Genre + Scene + Emotion
        () => `${randomWord(moodWords.genres)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.emotions)}`,
        // Multiple emotions
        () => `${randomWord(moodWords.emotions)} and ${randomWord(moodWords.emotions)}`,
        // Complex scene description
        () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`
    ];

    const template = randomWord(templates);
    return template();
}

function randomWord(wordArray) {
    return wordArray[Math.floor(Math.random() * wordArray.length)];
}

function createExampleChip(example, index) {
    const chip = document.createElement('button');
    chip.className = 'example-chip';
    chip.innerHTML = `<span class="example-emoji">${getRandomEmoji()}</span> ${example}`;

    chip.addEventListener('click', () => {
        useExample(example);
        // Add a subtle animation when clicked
        chip.style.transform = 'scale(0.95)';
        setTimeout(() => chip.style.transform = 'scale(1)', 150);
    });

    // Staggered animation
    chip.style.animationDelay = `${index * 0.1}s`;

    return chip;
}

function getRandomEmoji() {
    const emojis = ['🎵', '🎶', '🎧', '🎸', '🎹', '🥁', '🎷', '🎺', '🪕', '🎻', '✨', '🌟', '💫', '🔥', '💧', '🌊', '🍃', '🌙', '⭐', '⚡', '❤️', '🎉', '🌈', '🎨'];
    return randomWord(emojis);
}

// Use example
function useExample(text) {
    elements.moodInput.value = text;
    elements.moodInput.focus();
    // Auto-generate after a short delay to show the input change
    setTimeout(() => getRecommendations(), 300);
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
        console.log('Sending request to backend...');
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
        console.log('Received data from backend:', data);

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
    return section;
}

// Create track element - FIXED to handle artist_name properly
function createTrackElement(track, index) {
    // Use artist_name from backend response
    const artistName = track.artist_name || 'Unknown Artist';
    const songName = track.song_name || 'Unknown Track';
    const trackId = track.track_id || `track-${index}-${Date.now()}`;

    const isFavorite = state.favorites.some(fav => fav.track_id === trackId);

    const trackElement = document.createElement('div');
    trackElement.className = 'track-card';
    trackElement.innerHTML = `
        <div class="track-number">${index + 1}</div>
        <div class="track-info">
            <div class="track-title">${songName}</div>
            <div class="track-artist">${artistName}</div>
            ${track.why_it_matches ? `<div class="track-reason">${track.why_it_matches}</div>` : ''}
        </div>
        <div class="track-meta">
            <div class="similarity-score">
                ${track.match_percentage || Math.round((track.similarity_score || 0.8) * 100)}% match
            </div>
            <div class="track-actions">
                <button class="btn-action favorite ${isFavorite ? 'active' : ''}" onclick="toggleFavorite(${JSON.stringify({...track, track_id: trackId, artist_name: artistName, song_name: songName}).replace(/"/g, '&quot;')})">
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
    const trackId = track.track_id;
    const index = state.favorites.findIndex(fav => fav.track_id === trackId);

    if (index > -1) {
        state.favorites.splice(index, 1);
    } else {
        state.favorites.push(track);
    }

    localStorage.setItem('moodtune_favorites', JSON.stringify(state.favorites));
    updateFavoriteCount();

    // Re-render the current results to update heart icons
    if (state.recommendations.length > 0) {
        displayResults(state.recommendations, state.currentMood);
    }
}

// Update favorite count
function updateFavoriteCount() {
    elements.favoriteCount.textContent = state.favorites.length;
}

// Play preview
function playPreview(url, index) {
    alert(`Would play preview: ${url}\n\nTrack ${index + 1}`);
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
        const songName = track.song_name || track.name || 'Unknown Track';
        const artistName = track.artist_name || track.artist || 'Unknown Artist';
        favoritesText += `${index + 1}. ${songName} - ${artistName}\n`;
    });

    alert(favoritesText);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);
