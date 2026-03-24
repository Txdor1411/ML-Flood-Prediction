import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Dimensions,
  Image,
  KeyboardAvoidingView,
  PanResponder,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';
import { Asset } from 'expo-asset';
import * as NavigationBar from 'expo-navigation-bar';
import { WebView } from 'react-native-webview';

type PanelMode = 'routes' | 'chat' | 'alert';

type ChatMessage = {
  id: string;
  role: 'assistant' | 'user';
  text: string;
};

type PlaceResult = {
  name: string;
  display_name: string;
  lat: number;
  lon: number;
};

type RiskPoint = {
  risk_score: number;
  risk_band: 'low' | 'medium' | 'high' | 'extreme';
  recommendation: string;
  nearby_rivers: string[];
};

type RouteCard = {
  id: string;
  title: string;
  path: string;
  status: 'Clear' | 'Caution' | 'Priority';
  distance: string;
  eta: string;
};

type SituationSummary = {
  location_name: string;
  risk_score: number;
  risk_band: 'low' | 'medium' | 'high' | 'extreme';
  summary: string;
  nearby_rivers: string[];
};

const ROUTES: RouteCard[] = [
  {
    id: '1',
    title: 'Route 1',
    path: 'Ivesti -> Galati Sports Complex',
    status: 'Clear',
    distance: '18 km',
    eta: 'Est. 25 min',
  },
  {
    id: '2',
    title: 'Route 2',
    path: 'Targu Bujor -> Regional Shelter Beresti',
    status: 'Clear',
    distance: '14 km',
    eta: 'Est. 20 min',
  },
  {
    id: '3',
    title: 'Route 3',
    path: 'Tecuci -> Tecuci Community Center',
    status: 'Caution',
    distance: '8 km',
    eta: 'Est. 12 min',
  },
];

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL?.replace(/\/$/, '');
const LEGEND_COLORS = [
  '#2ca25f',
  '#46b56a',
  '#62c574',
  '#86d27a',
  '#abd97f',
  '#d1dd88',
  '#f0d97d',
  '#f8bf67',
  '#f29a53',
  '#e97040',
  '#d73027',
];

function formatRiskBandLabel(band: RiskPoint['risk_band'] | SituationSummary['risk_band']): string {
  if (band === 'low') {
    return 'Low';
  }
  if (band === 'medium') {
    return 'Moderate';
  }
  if (band === 'high') {
    return 'High';
  }
  return 'Extreme';
}

function getRiskColor(band: RiskPoint['risk_band'] | SituationSummary['risk_band']): string {
  if (band === 'low') {
    return '#76DD2F';
  }
  if (band === 'medium') {
    return '#E6C85C';
  }
  if (band === 'high') {
    return '#F39C4A';
  }
  return '#F17368';
}

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const PANEL_HEIGHT = Math.round(SCREEN_HEIGHT * 0.92);
const PANEL_PEEK_HEIGHT = Platform.OS === 'android' ? 64 : 40;
const PANEL_SNAP_TOP = 0;
const PANEL_SNAP_MID = Math.round(SCREEN_HEIGHT * 0.34);
const PANEL_SNAP_LOW = PANEL_HEIGHT - PANEL_PEEK_HEIGHT;
const FLOATING_BUTTONS_BOTTOM = 168;

type PanelSnap = 'top' | 'mid' | 'low';

export default function FloodMapScreen() {
  const [panelMode, setPanelMode] = useState<PanelMode>('routes');
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(true);
  const [mapUri, setMapUri] = useState<string | null>(null);
  const [routeCards, setRouteCards] = useState<RouteCard[]>(ROUTES);
  const [selectedLocationName, setSelectedLocationName] = useState('Romania (default view)');
  const [selectedLocationCoords, setSelectedLocationCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [searchInput, setSearchInput] = useState('');
  const [searchResults, setSearchResults] = useState<PlaceResult[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingLocationData, setIsLoadingLocationData] = useState(false);
  const [isLayersOpen, setIsLayersOpen] = useState(false);
  const [showRisk, setShowRisk] = useState(true);
  const [showRivers, setShowRivers] = useState(true);
  const [riskPoint, setRiskPoint] = useState<RiskPoint | null>(null);
  const [situationSummary, setSituationSummary] = useState<SituationSummary | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'a0',
      role: 'assistant',
      text: 'Ask me about flood risk, safety, or evacuation planning for your area.',
    },
  ]);
  const [isSending, setIsSending] = useState(false);
  const webViewRef = useRef<WebView>(null);
  const panelTranslateY = useRef(new Animated.Value(PANEL_SNAP_LOW)).current;
  const panelTranslateYRef = useRef(PANEL_SNAP_LOW);
  const panelDragStartRef = useRef(0);
  const [panelSnap, setPanelSnap] = useState<PanelSnap>('low');

  useEffect(() => {
    let isMounted = true;

    const loadLeafletMap = async () => {
      const asset = Asset.fromModule(require('@/assets/maps/romania_flood_risk_full.html'));
      await asset.downloadAsync();

      if (!isMounted) {
        return;
      }

      setMapUri(asset.localUri ?? asset.uri);
    };

    void loadLeafletMap();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'android') {
      return;
    }

    let isMounted = true;

    const keepAndroidSystemBarHidden = async () => {
      try {
        await NavigationBar.setPositionAsync('absolute');
        await NavigationBar.setBackgroundColorAsync('#00000000');
        await NavigationBar.setBehaviorAsync('overlay-swipe');
        await NavigationBar.setVisibilityAsync('hidden');
      } catch {
        // Ignore if this device/launcher does not allow immersive controls.
      }
    };

    void keepAndroidSystemBarHidden();

    const rehideInterval = setInterval(() => {
      if (!isMounted) {
        return;
      }

      void NavigationBar.setVisibilityAsync('hidden').catch(() => {
        // Ignore intermittent platform errors while re-hiding the nav bar.
      });
    }, 1200);

    return () => {
      isMounted = false;
      clearInterval(rehideInterval);
    };
  }, []);

  const panelTitle = useMemo(() => {
    if (panelMode === 'chat') {
      return 'FloodGuard AI';
    }
    if (panelMode === 'alert') {
      return 'Emergency Briefing';
    }
    return 'Evacuation Routes';
  }, [panelMode]);

  useEffect(() => {
    const listenerId = panelTranslateY.addListener(({ value }) => {
      panelTranslateYRef.current = value;
    });

    return () => {
      panelTranslateY.removeListener(listenerId);
    };
  }, [panelTranslateY]);

  const settlePanel = useCallback((target: PanelSnap) => {
    setPanelSnap(target);
    setIsPanelCollapsed(target === 'low');
    Animated.spring(panelTranslateY, {
      toValue: target === 'top' ? PANEL_SNAP_TOP : target === 'mid' ? PANEL_SNAP_MID : PANEL_SNAP_LOW,
      useNativeDriver: true,
      damping: 18,
      stiffness: 210,
      mass: 0.3,
    }).start();
  }, [panelTranslateY]);

  const panelPanResponder = useMemo(
    () =>
      PanResponder.create({
        onMoveShouldSetPanResponder: (_, gestureState) => Math.abs(gestureState.dy) > 3,
        onPanResponderGrant: () => {
          panelDragStartRef.current = panelTranslateYRef.current;
        },
        onPanResponderMove: (_, gestureState) => {
          const nextValue = Math.max(
            PANEL_SNAP_TOP,
            Math.min(PANEL_SNAP_LOW, panelDragStartRef.current + gestureState.dy)
          );
          panelTranslateY.setValue(nextValue);
        },
        onPanResponderRelease: (_, gestureState) => {
          const value = panelTranslateYRef.current;

          if (gestureState.vy > 0.7) {
            settlePanel(value < PANEL_SNAP_MID ? 'mid' : 'low');
            return;
          }

          if (gestureState.vy < -0.7) {
            settlePanel(value > PANEL_SNAP_MID ? 'mid' : 'top');
            return;
          }

          const distTop = Math.abs(value - PANEL_SNAP_TOP);
          const distMid = Math.abs(value - PANEL_SNAP_MID);
          const distLow = Math.abs(value - PANEL_SNAP_LOW);

          if (distTop <= distMid && distTop <= distLow) {
            settlePanel('top');
          } else if (distMid <= distLow) {
            settlePanel('mid');
          } else {
            settlePanel('low');
          }
        },
        onPanResponderTerminate: () => {
          const value = panelTranslateYRef.current;
          const distTop = Math.abs(value - PANEL_SNAP_TOP);
          const distMid = Math.abs(value - PANEL_SNAP_MID);
          const distLow = Math.abs(value - PANEL_SNAP_LOW);

          if (distTop <= distMid && distTop <= distLow) {
            settlePanel('top');
          } else if (distMid <= distLow) {
            settlePanel('mid');
          } else {
            settlePanel('low');
          }
        },
      }),
    [panelTranslateY, settlePanel]
  );

  const moveMapToLocation = useCallback((lat: number, lon: number, zoom = 11) => {
    const script = `
      (function() {
        var targetLat = ${lat};
        var targetLon = ${lon};
        var targetZoom = ${zoom};
        var mapRef = window.__floodguardMap || null;
        if (!mapRef) {
          for (var key in window) {
            if (Object.prototype.hasOwnProperty.call(window, key)) {
              var candidate = window[key];
              if (candidate && typeof candidate.setView === 'function' && typeof candidate.fitBounds === 'function') {
                mapRef = candidate;
                window.__floodguardMap = candidate;
                break;
              }
            }
          }
        }

        if (mapRef) {
          mapRef.setView([targetLat, targetLon], targetZoom, { animate: true });
          if (typeof L !== 'undefined') {
            if (window.__floodguardMarker) {
              window.__floodguardMarker.setLatLng([targetLat, targetLon]);
            } else {
              window.__floodguardMarker = L.circleMarker([targetLat, targetLon], {
                radius: 7,
                color: '#FFFFFF',
                weight: 2,
                fillColor: '#4EAFA8',
                fillOpacity: 0.95,
              }).addTo(mapRef);
            }
          }
        }
      })();
      true;
    `;
    webViewRef.current?.injectJavaScript(script);
  }, []);

  useEffect(() => {
    const script = `
      (function() {
        var mapRef = window.__floodguardMap || null;
        if (!mapRef) {
          for (var key in window) {
            if (Object.prototype.hasOwnProperty.call(window, key)) {
              var candidate = window[key];
              if (candidate && typeof candidate.setView === 'function' && typeof candidate.fitBounds === 'function') {
                mapRef = candidate;
                window.__floodguardMap = candidate;
                break;
              }
            }
          }
        }

        if (!mapRef) {
          return;
        }

        var riskLayer = window.__floodguardRiskLayer || null;
        var riversLayer = window.__floodguardRiversLayer || null;
        var shouldShowRisk = ${showRisk ? 'true' : 'false'};
        var shouldShowRivers = ${showRivers ? 'true' : 'false'};

        if (riskLayer) {
          if (shouldShowRisk) {
            mapRef.addLayer(riskLayer);
          } else {
            mapRef.removeLayer(riskLayer);
          }
        }

        if (riversLayer) {
          if (shouldShowRivers) {
            mapRef.addLayer(riversLayer);
          } else {
            mapRef.removeLayer(riversLayer);
          }
        }
      })();
      true;
    `;

    webViewRef.current?.injectJavaScript(script);
  }, [showRisk, showRivers]);

  const loadLocationContext = useCallback(async (location: PlaceResult) => {
    if (!API_BASE_URL) {
      setSearchError('Backend URL missing. Set EXPO_PUBLIC_API_BASE_URL in app env.');
      return;
    }

    setSelectedLocationName(location.name);
    setSelectedLocationCoords({ lat: location.lat, lon: location.lon });
    setIsLoadingLocationData(true);
    moveMapToLocation(location.lat, location.lon);

    try {
      const [riskResponse, routesResponse, summaryResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/risk/point`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: location.lat, lon: location.lon, rainfall_pct: 70 }),
        }),
        fetch(`${API_BASE_URL}/api/routes/cards`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: location.lat, lon: location.lon, rainfall_pct: 70 }),
        }),
        fetch(`${API_BASE_URL}/api/situation/summary`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat: location.lat,
            lon: location.lon,
            location_name: location.name,
            rainfall_pct: 70,
          }),
        }),
      ]);

      if (!riskResponse.ok || !routesResponse.ok || !summaryResponse.ok) {
        throw new Error('One or more location endpoints failed');
      }

      const riskPayload: RiskPoint = await riskResponse.json();
      const routesPayload: { routes: RouteCard[] } = await routesResponse.json();
      const summaryPayload: SituationSummary = await summaryResponse.json();

      setRiskPoint(riskPayload);
      setRouteCards(routesPayload.routes);
      setSituationSummary(summaryPayload);
      setPanelMode('routes');
      settlePanel('mid');
      setChatMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          text: `Updated context for ${location.name}: ${summaryPayload.summary}`,
        },
      ]);
    } catch {
      setSearchError('Failed to load flood data for this location. Please retry.');
    } finally {
      setIsLoadingLocationData(false);
    }
  }, [moveMapToLocation, settlePanel]);

  const runLocationSearch = useCallback(async () => {
    const query = searchInput.trim();
    if (!query) {
      setSearchResults([]);
      return;
    }
    if (!API_BASE_URL) {
      setSearchError('Backend URL missing. Set EXPO_PUBLIC_API_BASE_URL in app env.');
      return;
    }

    setIsSearching(true);
    setSearchError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/geocode/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 6 }),
      });

      if (!response.ok) {
        throw new Error('Search request failed');
      }

      const payload: { results: PlaceResult[] } = await response.json();
      setSearchResults(payload.results);
      if (payload.results.length === 0) {
        setSearchError('No Romanian location matches that query.');
      }
    } catch {
      setSearchError('Search failed. Check backend connection and retry.');
    } finally {
      setIsSearching(false);
    }
  }, [searchInput]);

  const sendMessage = async () => {
    if (isSending) {
      return;
    }

    const userText = chatInput.trim();
    if (!userText) {
      return;
    }

    if (!API_BASE_URL) {
      setChatMessages((prev) => [
        ...prev,
        { id: `u-${Date.now()}`, role: 'user', text: userText },
        {
          id: `a-${Date.now() + 1}`,
          role: 'assistant',
          text: 'Backend URL missing. Set EXPO_PUBLIC_API_BASE_URL in app env, then retry.',
        },
      ]);
      setChatInput('');
      return;
    }

    const userMessage: ChatMessage = { id: `u-${Date.now()}`, role: 'user', text: userText };
    const recentHistory = [...chatMessages, userMessage].slice(-8);

    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          location_name: selectedLocationName,
          messages: recentHistory.map((message) => ({
            role: message.role,
            content: message.text,
          })),
        }),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`Backend chat error ${response.status}: ${errorBody}`);
      }

      const data: { text?: string } = await response.json();
      const assistantText = data.text?.trim() ?? '';

      setChatMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          text:
            assistantText ||
            'I could not parse a response. Please try again or check your API settings.',
        },
      ]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          text: 'Connection failed. Please check backend and API key setup, then try again.',
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.mapBackground}>
        {mapUri ? (
          <WebView
            ref={webViewRef}
            source={{ uri: mapUri }}
            style={styles.webView}
            originWhitelist={['*']}
            allowFileAccess
            allowUniversalAccessFromFileURLs
            javaScriptEnabled
            domStorageEnabled
            startInLoadingState
            renderLoading={() => (
              <View style={styles.webViewLoader}>
                <ActivityIndicator size="large" color="#FFFFFF" />
              </View>
            )}
          />
        ) : (
          <View style={styles.webViewLoader}>
            <ActivityIndicator size="large" color="#FFFFFF" />
          </View>
        )}
        <View style={styles.mapTint} pointerEvents="none" />

        <SafeAreaView style={styles.safeArea} pointerEvents="box-none">
          <View style={styles.headerSection}>
            <View style={styles.searchShell}>
              <View style={styles.searchInputRow}>
                <TextInput
                  placeholder="Search a location in Romania"
                  placeholderTextColor="#BEC7E5"
                  style={styles.searchInput}
                  value={searchInput}
                  onChangeText={setSearchInput}
                  returnKeyType="search"
                  onSubmitEditing={runLocationSearch}
                />
                <Pressable style={styles.searchButton} onPress={runLocationSearch} disabled={isSearching}>
                  {isSearching ? (
                    <ActivityIndicator size="small" color="#FFFFFF" />
                  ) : (
                    <Feather name="search" size={18} color="#FFFFFF" />
                  )}
                </Pressable>
              </View>
              {searchError ? <Text style={styles.searchErrorText}>{searchError}</Text> : null}
              {searchResults.length > 0 && (
                <ScrollView style={styles.searchResultsWrap} keyboardShouldPersistTaps="handled">
                  {searchResults.map((result) => (
                    <Pressable
                      key={`${result.lat}-${result.lon}`}
                      style={styles.searchResultItem}
                      onPress={() => {
                        setSearchResults([]);
                        setSearchInput(result.name);
                        setSearchError(null);
                        void loadLocationContext(result);
                      }}>
                      <Text style={styles.searchResultTitle}>{result.name}</Text>
                      <Text style={styles.searchResultSubtitle} numberOfLines={1}>
                        {result.display_name}
                      </Text>
                    </Pressable>
                  ))}
                </ScrollView>
              )}
            </View>

            <View style={styles.topRow}>
              <View style={styles.logoContainer}>
                <Image
                  source={require('@/assets/images/Flood Guard png logo white.png')}
                  style={styles.logoImage}
                  resizeMode="contain"
                />
              </View>
              <Pressable
                style={styles.layersButton}
                onPress={() => setIsLayersOpen(!isLayersOpen)}>
                <MaterialCommunityIcons name="menu" size={24} color="#FFFFFF" />
              </Pressable>
            </View>
          </View>

          <View style={[styles.rightButtonsColumn, { bottom: FLOATING_BUTTONS_BOTTOM }]}> 
            <RoundActionButton
              active={panelMode === 'chat'}
              icon={<MaterialCommunityIcons name="robot-outline" size={24} color="#FFFFFF" />}
              activeColor="#4EAFA8"
              idleColor="#4D3E95"
              onPress={() => setPanelMode('chat')}
            />
            <RoundActionButton
              active={panelMode === 'routes'}
              icon={<MaterialCommunityIcons name="clipboard-text-outline" size={24} color="#FFFFFF" />}
              activeColor="#4EAFA8"
              idleColor="#4D3E95"
              onPress={() => setPanelMode('routes')}
            />
            <RoundActionButton
              active={panelMode === 'alert'}
              icon={<Feather name="volume-2" size={24} color="#FFFFFF" />}
              activeColor="#4EAFA8"
              idleColor="#4D3E95"
              onPress={() => setPanelMode('alert')}
            />
          </View>

          <KeyboardAvoidingView
            style={styles.keyboardLayer}
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            keyboardVerticalOffset={8}>
            <Animated.View
              style={[
                styles.bottomPanel,
                { transform: [{ translateY: panelTranslateY }] },
              ]}>
            <View style={styles.panelDragTouchZone} {...panelPanResponder.panHandlers} />
            <View style={styles.panelDragArea} pointerEvents="none">
              <View style={styles.handle} />
            </View>
            {!isPanelCollapsed && <Text style={styles.panelTitle}>{panelTitle}</Text>}

            {!isPanelCollapsed && panelMode === 'routes' && (
              <View style={styles.routesList}>
                <Text style={styles.locationContextText}>Selected: {selectedLocationName}</Text>
                {isLoadingLocationData ? (
                  <ActivityIndicator size="small" color="#FFFFFF" style={styles.locationLoadingSpinner} />
                ) : null}
                {routeCards.map((route, index) => {
                  const isHighlighted = index === 1;
                  return (
                    <View
                      key={route.id}
                      style={[styles.routeCard, isHighlighted && styles.routeCardHighlighted]}>
                      <Text style={styles.routeTitle}>{route.title}</Text>
                      <Text style={styles.routePath}>{route.path}</Text>
                      <Text
                        style={[
                          styles.routeStatus,
                          route.status === 'Caution' && styles.statusWarn,
                          route.status === 'Priority' && styles.statusPriority,
                        ]}>
                        {route.status}
                      </Text>
                      <View>
                        <Text style={styles.routeDistance}>{route.distance}</Text>
                        <Text style={styles.routeEta}>{route.eta}</Text>
                      </View>
                    </View>
                  );
                })}
              </View>
            )}

            {!isPanelCollapsed && panelMode === 'chat' && (
              <View style={styles.chatWrap}>
                {chatMessages.slice(-4).map((message) => (
                  <View
                    key={message.id}
                    style={[message.role === 'user' ? styles.questionBubble : styles.answerRow]}>
                    {message.role === 'assistant' && (
                      <MaterialCommunityIcons name="robot-outline" size={20} color="#FFFFFF" />
                    )}
                    <Text style={message.role === 'user' ? styles.questionText : styles.answerText}>
                      {message.text}
                    </Text>
                  </View>
                ))}

                <View style={styles.chatInputRow}>
                  <TextInput
                    placeholder="Message FloodGuard AI..."
                    placeholderTextColor="#D0D6E8"
                    style={styles.chatInput}
                    value={chatInput}
                    onChangeText={setChatInput}
                    editable={!isSending}
                    onSubmitEditing={sendMessage}
                  />
                  <Pressable disabled>
                    <Feather name="mic" size={22} color="#FFFFFF" />
                  </Pressable>
                  <Pressable onPress={sendMessage} disabled={isSending}>
                    {isSending ? (
                      <ActivityIndicator size="small" color="#FFFFFF" />
                    ) : (
                      <Feather name="send" size={22} color="#FFFFFF" />
                    )}
                  </Pressable>
                </View>
              </View>
            )}

            {!isPanelCollapsed && panelMode === 'alert' && (
              <View style={styles.alertWrap}>
                <View style={styles.alertHeader}>
                  <Feather
                    name="alert-triangle"
                    size={24}
                    color={riskPoint ? getRiskColor(riskPoint.risk_band) : '#F17368'}
                  />
                  <Text style={styles.alertText}>
                    {situationSummary?.summary ??
                      'Select a location to load flood conditions and recommendations for that area.'}
                  </Text>
                </View>
                {riskPoint && (
                  <Text style={styles.alertRiskLine}>
                    Risk: {formatRiskBandLabel(riskPoint.risk_band)} ({riskPoint.risk_score.toFixed(2)})
                  </Text>
                )}
                {riskPoint?.nearby_rivers?.length ? (
                  <Text style={styles.alertRiversLine}>
                    Nearby rivers: {riskPoint.nearby_rivers.join(', ')}
                  </Text>
                ) : null}
                <Text style={styles.alertListTitle}>Take action now:</Text>
                <Text style={styles.alertBullet}>- Keep documents and an emergency bag ready.</Text>
                <Text style={styles.alertBullet}>- If you are near a river, prepare for evacuation.</Text>
                <Text style={styles.alertBullet}>- Follow local authority and emergency updates.</Text>
                <Text style={styles.alertBullet}>- Avoid crossing flooded roads or underpasses.</Text>

                <View style={styles.alertActions}>
                  <Pressable style={styles.secondaryButton} onPress={() => setPanelMode('chat')}>
                    <MaterialCommunityIcons name="robot-outline" size={20} color="#DBE2FF" />
                  </Pressable>
                  <Pressable
                    style={styles.primaryButton}
                    onPress={() => {
                      setPanelMode('routes');
                      settlePanel('mid');
                    }}>
                    <Text style={styles.primaryButtonText}>View evacuation routes</Text>
                  </Pressable>
                </View>
              </View>
            )}
            </Animated.View>
          </KeyboardAvoidingView>

          {isLayersOpen && (
            <View style={styles.layersPanel}>
              <View style={styles.layersPanelContent}>
                <View style={styles.layersPanelHeader}>
                  <Text style={styles.layersPanelTitle}>Map Layers</Text>
                  <Pressable onPress={() => setIsLayersOpen(false)}>
                    <Feather name="x" size={24} color="#FFFFFF" />
                  </Pressable>
                </View>

                <View style={styles.layersPanelSection}>
                  <Text style={styles.layersSectionTitle}>Map Layers</Text>
                  <View style={styles.layerToggle}>
                    <Text style={styles.layerToggleLabel}>Show risk colors</Text>
                    <Pressable
                      onPress={() => setShowRisk(!showRisk)}
                      style={[styles.toggleSwitch, showRisk && styles.toggleSwitchActive]}>
                      <View style={[styles.toggleThumb, showRisk && styles.toggleThumbActive]} />
                    </Pressable>
                  </View>
                  <View style={styles.layerToggle}>
                    <Text style={styles.layerToggleLabel}>Show rivers</Text>
                    <Pressable
                      onPress={() => setShowRivers(!showRivers)}
                      style={[styles.toggleSwitch, showRivers && styles.toggleSwitchActive]}>
                      <View style={[styles.toggleThumb, showRivers && styles.toggleThumbActive]} />
                    </Pressable>
                  </View>
                </View>

                <View style={styles.layersPanelSection}>
                  <Text style={styles.layersSectionTitle}>Flood Risk Legend</Text>
                  <View style={styles.legendContainer}>
                    <Text style={styles.legendCaption}>Relative flood risk (0.0 to 1.0)</Text>
                    <View style={styles.legendBar}>
                      {LEGEND_COLORS.map((color, index) => (
                        <View key={`${color}-${index}`} style={[styles.legendBarSegment, { backgroundColor: color, flex: 1 }]} />
                      ))}
                    </View>
                    <View style={styles.legendTicks}>
                      <Text style={styles.legendTickText}>0.0</Text>
                      <Text style={styles.legendTickText}>0.5</Text>
                      <Text style={styles.legendTickText}>1.0</Text>
                    </View>
                  </View>
                </View>
              </View>
            </View>
          )}
        </SafeAreaView>
      </View>
    </View>
  );
}

function RoundActionButton({
  icon,
  active,
  activeColor,
  idleColor,
  onPress,
}: {
  icon: React.ReactNode;
  active: boolean;
  activeColor: string;
  idleColor: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.roundAction,
        { backgroundColor: active ? activeColor : idleColor },
        pressed && styles.roundActionPressed,
      ]}>
      {icon}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B1230',
  },
  mapBackground: {
    ...StyleSheet.absoluteFillObject,
  },
  webView: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#0A1029',
  },
  webViewLoader: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0A1029',
  },
  mapTint: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(33, 24, 42, 0.12)',
  },
  safeArea: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-start',
  },
  headerSection: {
    paddingHorizontal: 0,
    paddingTop: 0,
    zIndex: 40,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 0,
    marginTop: 8,
    position: 'relative',
  },
  searchShell: {
    marginTop: 12,
    marginHorizontal: 12,
    padding: 10,
    borderRadius: 14,
    backgroundColor: 'rgba(15, 29, 65, 0.84)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.16)',
    zIndex: 35,
  },
  searchInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  searchInput: {
    flex: 1,
    minHeight: 42,
    borderRadius: 12,
    paddingHorizontal: 12,
    color: '#FFFFFF',
    fontSize: 15,
    backgroundColor: 'rgba(102, 126, 189, 0.32)',
  },
  searchButton: {
    width: 42,
    height: 42,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2E6588',
  },
  searchErrorText: {
    color: '#F7B3AB',
    marginTop: 7,
    fontSize: 12,
    fontWeight: '500',
  },
  searchResultsWrap: {
    maxHeight: 168,
    marginTop: 8,
  },
  searchResultItem: {
    backgroundColor: 'rgba(255,255,255,0.10)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginBottom: 6,
  },
  searchResultTitle: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  searchResultSubtitle: {
    marginTop: 2,
    color: '#CFD7F4',
    fontSize: 12,
  },
  logoContainer: {
    width: 200,
    height: 70,
    marginLeft: -40,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  logoImage: {
    width: '138%',
    height: '100%',
    marginLeft: -52,
    tintColor: '#000000',
  },
  rightButtonsColumn: {
    position: 'absolute',
    right: 20,
    zIndex: 20,
    gap: 12,
    alignItems: 'center',
  },
  keyboardLayer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
  },
  roundAction: {
    width: 60,
    height: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000000',
    shadowOpacity: 0.28,
    shadowRadius: 7,
    shadowOffset: { width: 0, height: 5 },
    elevation: 8,
  },
  roundActionPressed: {
    opacity: 0.84,
  },
  bottomPanel: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: PANEL_HEIGHT,
    backgroundColor: '#111F56',
    borderTopLeftRadius: 34,
    borderTopRightRadius: 34,
    paddingHorizontal: 18,
    paddingTop: 14,
    paddingBottom: Platform.OS === 'ios' ? 34 : 22,
    opacity: 1,
  },
  panelDragArea: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
    paddingVertical: 18,
  },
  panelDragTouchZone: {
    position: 'absolute',
    top: -24,
    left: 0,
    right: 0,
    height: 170,
    zIndex: 5,
  },
  handle: {
    width: 96,
    height: 6,
    borderRadius: 999,
    backgroundColor: '#D9DDE7',
    alignSelf: 'center',
    opacity: 0.95,
  },
  panelTitle: {
    color: '#FFFFFF',
    fontSize: 30,
    fontWeight: '700',
    textAlign: 'center',
    marginTop: 4,
    marginBottom: 18,
  },
  routesList: {
    gap: 10,
  },
  locationContextText: {
    color: '#DCE4FF',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 2,
  },
  locationLoadingSpinner: {
    marginBottom: 4,
    alignSelf: 'flex-start',
  },
  routeCard: {
    backgroundColor: 'transparent',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  routeCardHighlighted: {
    backgroundColor: '#2E6588',
  },
  routeTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '500',
    minWidth: 84,
  },
  routePath: {
    color: '#DCE4FF',
    fontSize: 12,
    fontWeight: '600',
    flex: 1,
  },
  routeStatus: {
    color: '#76DD2F',
    fontSize: 18,
    fontWeight: '700',
    minWidth: 68,
  },
  statusWarn: {
    color: '#D5852A',
  },
  statusPriority: {
    color: '#F17368',
  },
  routeDistance: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
    textAlign: 'right',
  },
  routeEta: {
    color: '#C9D3EA',
    fontSize: 11,
    fontWeight: '500',
    textAlign: 'right',
  },
  chatWrap: {
    gap: 14,
  },
  questionBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#677091',
    borderRadius: 18,
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  questionText: {
    color: '#F5F7FF',
    fontSize: 16,
    fontWeight: '500',
  },
  answerRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
    backgroundColor: '#25326788',
    borderRadius: 16,
    padding: 10,
  },
  answerText: {
    flex: 1,
    color: '#F2F5FF',
    fontSize: 19,
    lineHeight: 26,
    fontWeight: '500',
  },
  chatInputRow: {
    marginTop: 4,
    backgroundColor: '#6F7895',
    borderRadius: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 14,
    minHeight: 56,
  },
  chatInput: {
    flex: 1,
    color: '#FFFFFF',
    fontSize: 18,
  },
  alertWrap: {
    gap: 8,
  },
  alertHeader: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  alertText: {
    flex: 1,
    color: '#F2F5FF',
    fontSize: 16,
    lineHeight: 24,
    fontWeight: '500',
  },
  alertListTitle: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '700',
    marginTop: 2,
  },
  alertRiskLine: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    marginTop: 6,
  },
  alertRiversLine: {
    color: '#D6DDF5',
    fontSize: 13,
    marginTop: 2,
  },
  alertBullet: {
    color: '#F2F5FF',
    fontSize: 15,
    lineHeight: 22,
  },
  alertActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginTop: 14,
  },
  secondaryButton: {
    borderWidth: 2,
    borderColor: '#D8E0FF55',
    borderRadius: 16,
    width: 58,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButton: {
    borderWidth: 2,
    borderColor: '#CAD6FF66',
    borderRadius: 16,
    flex: 1,
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 18,
  },
  layersButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    backgroundColor: 'rgba(102, 126, 189, 0.32)',
    marginRight: 12,
  },
  layersPanel: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    zIndex: 100,
    justifyContent: 'flex-start',
    paddingTop: 60,
  },
  layersPanelContent: {
    marginHorizontal: 12,
    backgroundColor: 'rgba(15, 29, 65, 0.95)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.16)',
  },
  layersPanelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  layersPanelTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
  layersPanelSection: {
    gap: 12,
  },
  layersSectionTitle: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  layersNote: {
    color: '#CFD7F4',
    fontSize: 13,
    fontStyle: 'italic',
  },
  layerToggle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 10,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    marginBottom: 8,
  },
  layerToggleLabel: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '500',
  },
  toggleSwitch: {
    width: 48,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(255,255,255,0.2)',
    padding: 2,
    justifyContent: 'center',
  },
  toggleSwitchActive: {
    backgroundColor: '#4EAFA8',
  },
  toggleThumb: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    alignSelf: 'flex-start',
  },
  toggleThumbActive: {
    alignSelf: 'flex-end',
  },
  legendContainer: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
  },
  legendCaption: {
    color: '#F1F3F5',
    fontSize: 12,
    marginBottom: 8,
  },
  legendBar: {
    width: '100%',
    height: 24,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    marginBottom: 8,
    backgroundColor: '#2ca25f',
    flexDirection: 'row',
    overflow: 'hidden',
  },
  legendBarSegment: {
    height: '100%',
  },
  legendTicks: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 4,
  },
  legendTickText: {
    color: '#CFD7F4',
    fontSize: 11,
  },
});
