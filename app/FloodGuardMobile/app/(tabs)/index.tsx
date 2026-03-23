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

const ROUTES = [
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
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'a0',
      role: 'assistant',
      text: 'Ask me about flood risk, safety, or evacuation planning for your area.',
    },
  ]);
  const [isSending, setIsSending] = useState(false);
  const panelTranslateY = useRef(new Animated.Value(PANEL_SNAP_LOW)).current;
  const panelTranslateYRef = useRef(PANEL_SNAP_LOW);
  const panelDragStartRef = useRef(0);
  const [panelSnap, setPanelSnap] = useState<PanelSnap>('low');

  useEffect(() => {
    let isMounted = true;

    const loadLeafletMap = async () => {
      const asset = Asset.fromModule(require('@/assets/maps/cluj_flood_risk.html'));
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

  const sendMessage = async () => {
    if (isSending) {
      return;
    }

    const userText = chatInput.trim();
    if (!userText) {
      return;
    }

    const apiKey = process.env.EXPO_PUBLIC_OPENAI_API_KEY;
    if (!apiKey) {
      setChatMessages((prev) => [
        ...prev,
        { id: `u-${Date.now()}`, role: 'user', text: userText },
        {
          id: `a-${Date.now() + 1}`,
          role: 'assistant',
          text: 'OpenAI key missing. Set EXPO_PUBLIC_OPENAI_API_KEY in your app env, then retry.',
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
      const response = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4.1-mini',
          input: [
            {
              role: 'system',
              content:
                'You are FloodGuard AI. Give concise, practical flood safety guidance and evacuation advice.',
            },
            ...recentHistory.map((message) => ({
              role: message.role,
              content: message.text,
            })),
          ],
        }),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`OpenAI API error ${response.status}: ${errorBody}`);
      }

      const data = await response.json();
      const assistantText = extractAssistantText(data);

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
          text: 'Connection failed. Please check network and API key, then try again.',
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
          <View style={styles.topRow}>
            <View style={styles.logoContainer}>
              <Image
                source={require('@/assets/images/Flood Guard png logo white.png')}
                style={styles.logoImage}
                resizeMode="contain"
              />
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
                {ROUTES.map((route, index) => {
                  const isHighlighted = index === 1;
                  return (
                    <View
                      key={route.id}
                      style={[styles.routeCard, isHighlighted && styles.routeCardHighlighted]}>
                      <Text style={styles.routeTitle}>{route.title}</Text>
                      <Text style={styles.routePath}>{route.path}</Text>
                      <Text style={[styles.routeStatus, route.status === 'Caution' && styles.statusWarn]}>
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
                  <Feather name="alert-triangle" size={24} color="#F17368" />
                  <Text style={styles.alertText}>
                    Heavy rain is forecast over the next hours and nearby river levels are rising.
                    There is high flood risk for low-lying zones.
                  </Text>
                </View>
                <Text style={styles.alertListTitle}>Take action now:</Text>
                <Text style={styles.alertBullet}>- Keep documents and an emergency bag ready.</Text>
                <Text style={styles.alertBullet}>- If you are near a river, prepare for evacuation.</Text>
                <Text style={styles.alertBullet}>- Follow local authority and emergency updates.</Text>
                <Text style={styles.alertBullet}>- Avoid crossing flooded roads or underpasses.</Text>

                <View style={styles.alertActions}>
                  <Pressable style={styles.secondaryButton}>
                    <MaterialCommunityIcons name="robot-outline" size={20} color="#DBE2FF" />
                  </Pressable>
                  <Pressable style={styles.primaryButton}>
                    <Text style={styles.primaryButtonText}>View evacuation routes</Text>
                  </Pressable>
                </View>
              </View>
            )}
            </Animated.View>
          </KeyboardAvoidingView>
        </SafeAreaView>
      </View>
    </View>
  );
}

function extractAssistantText(payload: any): string {
  if (typeof payload?.output_text === 'string' && payload.output_text.length > 0) {
    return payload.output_text;
  }

  const chunks = payload?.output?.flatMap?.((entry: any) =>
    (entry?.content ?? [])
      .filter((part: any) => part?.type === 'output_text' || part?.type === 'text')
      .map((part: any) => part?.text ?? part?.output_text ?? '')
  );

  return Array.isArray(chunks) ? chunks.join('\n').trim() : '';
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
    justifyContent: 'space-between',
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingHorizontal: 0,
    marginTop: 8,
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
    top: 0,
    left: 0,
    right: 0,
    height: 120,
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
});
