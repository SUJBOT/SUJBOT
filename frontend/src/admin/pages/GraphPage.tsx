/**
 * Interactive Knowledge Graph Visualization
 *
 * Uses Sigma.js (WebGL) + graphology for rendering up to 500 nodes.
 * Filters by document, community, and entity type.
 * Search highlights matching nodes and dims the rest.
 */

import { useEffect, useState, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Button,
  TextField,
  Autocomplete,
  Chip,
  Alert,
  Tabs,
  Tab,
  List,
  ListItemButton,
  ListItemText,
  Divider,
  Paper,
} from '@mui/material';
import { Network, Search, Download } from 'lucide-react';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import {
  SigmaContainer,
  useLoadGraph,
  useRegisterEvents,
  useSigma,
  useSetSettings,
} from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';

import { getApiBaseUrl } from '../dataProvider';

const API_BASE_URL = getApiBaseUrl();

// ─── Types ──────────────────────────────────────────────────────────────────

interface GraphOverview {
  stats: { entities: number; relationships: number; communities: number };
  documents: { document_id: string; display_name: string; entity_count: number }[];
  communities: { community_id: number; title: string; entity_count: number }[];
  entity_types: { type: string; count: number }[];
}

interface GraphNode {
  id: number;
  name: string;
  type: string;
  description: string | null;
  document_id: string;
  community_id: number | null;
}

interface GraphEdge {
  source: number;
  target: number;
  type: string;
  description: string | null;
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_matching: number;
  limited: boolean;
}

// ─── Color palette by entity type ───────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  REGULATION: '#3b82f6',
  ORGANIZATION: '#22c55e',
  PERSON: '#f59e0b',
  LOCATION: '#ef4444',
  STANDARD: '#8b5cf6',
  SECTION: '#06b6d4',
  CONCEPT: '#ec4899',
  DOCUMENT: '#64748b',
  PROCEDURE: '#14b8a6',
  REQUIREMENT: '#f97316',
  EQUIPMENT: '#84cc16',
  MATERIAL: '#a855f7',
  EVENT: '#e11d48',
  METRIC: '#0ea5e9',
  ROLE: '#d946ef',
};
const DEFAULT_COLOR = '#94a3b8';

function getTypeColor(type: string): string {
  return TYPE_COLORS[type] || DEFAULT_COLOR;
}

// ─── Tooltip component ─────────────────────────────────────────────────────

interface TooltipState {
  x: number;
  y: number;
  type: 'node' | 'edge';
  data: GraphNode | GraphEdge;
}

function GraphTooltip({ tooltip }: { tooltip: TooltipState | null }) {
  if (!tooltip) return null;

  return (
    <Paper
      elevation={4}
      sx={{
        position: 'fixed',
        left: tooltip.x + 12,
        top: tooltip.y + 12,
        p: 1.5,
        maxWidth: 320,
        zIndex: 1500,
        pointerEvents: 'none',
        fontSize: '0.85rem',
      }}
    >
      {tooltip.type === 'node' ? (
        <>
          <Typography variant="subtitle2" fontWeight={700}>
            {(tooltip.data as GraphNode).name}
          </Typography>
          <Chip
            label={(tooltip.data as GraphNode).type}
            size="small"
            sx={{
              mt: 0.5,
              mb: 0.5,
              backgroundColor: getTypeColor((tooltip.data as GraphNode).type),
              color: '#fff',
              fontSize: '0.75rem',
            }}
          />
          {(tooltip.data as GraphNode).description && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              {(tooltip.data as GraphNode).description!.slice(0, 200)}
              {(tooltip.data as GraphNode).description!.length > 200 ? '...' : ''}
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
            {(tooltip.data as GraphNode).document_id}
          </Typography>
        </>
      ) : (
        <>
          <Typography variant="subtitle2" fontWeight={700}>
            {(tooltip.data as GraphEdge).type}
          </Typography>
          {(tooltip.data as GraphEdge).description && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              {(tooltip.data as GraphEdge).description!.slice(0, 200)}
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary">
            weight: {(tooltip.data as GraphEdge).weight.toFixed(1)}
          </Typography>
        </>
      )}
    </Paper>
  );
}

// ─── Sigma inner components ─────────────────────────────────────────────────

/** Loads graph data into graphology + runs ForceAtlas2 layout */
function GraphLoader({
  graphData,
  onNodeMap,
}: {
  graphData: GraphData;
  onNodeMap: (map: Map<string, GraphNode>) => void;
}) {
  const loadGraph = useLoadGraph();

  useEffect(() => {
    const graph = new Graph();
    const nodeMap = new Map<string, GraphNode>();

    for (const node of graphData.nodes) {
      const key = String(node.id);
      graph.addNode(key, {
        label: node.name,
        size: 6,
        color: getTypeColor(node.type),
        x: Math.random() * 100,
        y: Math.random() * 100,
      });
      nodeMap.set(key, node);
    }

    const nodeSet = new Set(graphData.nodes.map((n) => n.id));
    const addedEdges = new Set<string>();
    for (const edge of graphData.edges) {
      if (!nodeSet.has(edge.source) || !nodeSet.has(edge.target)) continue;
      const src = String(edge.source);
      const tgt = String(edge.target);
      if (src === tgt) continue;
      // Deduplicate: one edge per node pair (collapse multiple relationship types)
      const pairKey = src < tgt ? `${src}-${tgt}` : `${tgt}-${src}`;
      if (addedEdges.has(pairKey)) continue;
      addedEdges.add(pairKey);
      graph.addEdgeWithKey(pairKey, src, tgt, {
        label: edge.type,
        weight: edge.weight,
        size: 1,
        color: '#cbd5e1',
      });
    }

    // Scale node size by degree
    graph.forEachNode((key) => {
      const degree = graph.degree(key);
      graph.setNodeAttribute(key, 'size', Math.max(4, Math.min(20, 4 + degree * 1.5)));
    });

    // Run ForceAtlas2 layout (synchronous, ~100ms for 500 nodes)
    forceAtlas2.assign(graph, {
      iterations: 100,
      settings: {
        gravity: 1,
        scalingRatio: 10,
        barnesHutOptimize: graphData.nodes.length > 100,
        strongGravityMode: true,
      },
    });

    loadGraph(graph);
    onNodeMap(nodeMap);
  }, [graphData, loadGraph, onNodeMap]);

  return null;
}

/** Handles hover events for tooltips */
function GraphEvents({
  nodeMap,
  graphData,
  setTooltip,
}: {
  nodeMap: Map<string, GraphNode>;
  graphData: GraphData;
  setTooltip: (t: TooltipState | null) => void;
}) {
  const registerEvents = useRegisterEvents();
  const sigma = useSigma();

  useEffect(() => {
    // Build edge lookup
    const edgeLookup = new Map<string, GraphEdge>();
    for (const edge of graphData.edges) {
      edgeLookup.set(`${edge.source}-${edge.target}-${edge.type}`, edge);
    }

    const getClientCoords = (original: MouseEvent | TouchEvent) => {
      if ('clientX' in original) return { x: original.clientX, y: original.clientY };
      const touch = (original as TouchEvent).touches[0];
      return touch ? { x: touch.clientX, y: touch.clientY } : { x: 0, y: 0 };
    };

    registerEvents({
      enterNode: (event) => {
        const coords = getClientCoords(event.event.original);
        const nodeData = nodeMap.get(event.node);
        if (nodeData) {
          setTooltip({ x: coords.x, y: coords.y, type: 'node', data: nodeData });
        }
      },
      leaveNode: () => setTooltip(null),
      enterEdge: (event) => {
        const coords = getClientCoords(event.event.original);
        const edgeData = edgeLookup.get(event.edge);
        if (edgeData) {
          setTooltip({ x: coords.x, y: coords.y, type: 'edge', data: edgeData });
        }
      },
      leaveEdge: () => setTooltip(null),
    });
  }, [nodeMap, graphData, registerEvents, sigma, setTooltip]);

  return null;
}

/** Handles search highlighting by dimming non-matching nodes */
function SearchHighlighter({ searchQuery }: { searchQuery: string }) {
  const sigma = useSigma();
  const setSettings = useSetSettings();

  useEffect(() => {
    if (!searchQuery.trim()) {
      // Reset — show all nodes at full opacity
      setSettings({
        nodeReducer: undefined,
        edgeReducer: undefined,
      });
      return;
    }

    const query = searchQuery.toLowerCase();
    const graph = sigma.getGraph();
    const matchingNodes = new Set<string>();

    graph.forEachNode((key, attrs) => {
      if ((attrs.label as string)?.toLowerCase().includes(query)) {
        matchingNodes.add(key);
      }
    });

    setSettings({
      nodeReducer: (node, data) => {
        if (matchingNodes.size === 0) return data;
        if (matchingNodes.has(node)) {
          return { ...data, highlighted: true, zIndex: 1 };
        }
        return { ...data, color: '#e2e8f0', label: '', zIndex: 0 };
      },
      edgeReducer: (_edge, data) => {
        if (matchingNodes.size === 0) return data;
        return { ...data, hidden: true };
      },
    });

    // Auto-focus camera on matching nodes
    if (matchingNodes.size > 0 && matchingNodes.size <= 50) {
      const positions = Array.from(matchingNodes).map((key) => ({
        x: graph.getNodeAttribute(key, 'x') as number,
        y: graph.getNodeAttribute(key, 'y') as number,
      }));
      const cx = positions.reduce((s, p) => s + p.x, 0) / positions.length;
      const cy = positions.reduce((s, p) => s + p.y, 0) / positions.length;
      sigma.getCamera().animate({ x: cx, y: cy, ratio: 0.5 }, { duration: 300 });
    }
  }, [searchQuery, sigma, setSettings]);

  return null;
}

// ─── Main page component ────────────────────────────────────────────────────

export function GraphPage() {
  const { t } = useTranslation();

  // Overview data (filter options)
  const [overview, setOverview] = useState<GraphOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);

  // Filter selections
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [selectedCommunities, setSelectedCommunities] = useState<number[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);

  // Graph data
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  // UI state
  const [searchQuery, setSearchQuery] = useState('');
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [sidebarTab, setSidebarTab] = useState(0);
  const [nodeMap, setNodeMap] = useState<Map<string, GraphNode>>(new Map());

  // Fetch overview on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/admin/graph/overview`, {
          credentials: 'include',
          headers: { Accept: 'application/json' },
        });
        if (!res.ok) throw new Error('Failed to fetch graph overview');
        setOverview(await res.json());
      } catch (err) {
        setGraphError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setOverviewLoading(false);
      }
    })();
  }, []);

  // Load graph data
  const loadGraph = useCallback(async () => {
    if (!selectedDocs.length && !selectedCommunities.length && !selectedTypes.length) return;

    setGraphLoading(true);
    setGraphError(null);
    setGraphData(null);
    setSearchQuery('');

    const params = new URLSearchParams();
    if (selectedDocs.length) params.set('document_ids', selectedDocs.join(','));
    if (selectedCommunities.length) params.set('community_ids', selectedCommunities.join(','));
    if (selectedTypes.length) params.set('entity_types', selectedTypes.join(','));

    try {
      const res = await fetch(`${API_BASE_URL}/admin/graph/data?${params}`, {
        credentials: 'include',
        headers: { Accept: 'application/json' },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Failed to fetch graph data');
      }
      setGraphData(await res.json());
    } catch (err) {
      setGraphError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setGraphLoading(false);
    }
  }, [selectedDocs, selectedCommunities, selectedTypes]);

  const handleNodeMapUpdate = useCallback((map: Map<string, GraphNode>) => {
    setNodeMap(map);
  }, []);

  // Group nodes by type for sidebar tab
  const nodesByType = useMemo(() => {
    if (!graphData) return new Map<string, GraphNode[]>();
    const map = new Map<string, GraphNode[]>();
    for (const node of graphData.nodes) {
      const list = map.get(node.type) || [];
      list.push(node);
      map.set(node.type, list);
    }
    return map;
  }, [graphData]);

  // Community list for sidebar tab
  const visibleCommunities = useMemo(() => {
    if (!graphData || !overview) return [];
    const communityIds = new Set(
      graphData.nodes.map((n) => n.community_id).filter((id): id is number => id != null),
    );
    return overview.communities.filter((c) => communityIds.has(c.community_id));
  }, [graphData, overview]);

  const hasFilters = selectedDocs.length > 0 || selectedCommunities.length > 0 || selectedTypes.length > 0;

  if (overviewLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3} sx={{ height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box mb={2}>
        <Box display="flex" alignItems="center" gap={1} mb={0.5}>
          <Network size={24} />
          <Typography variant="h5">{t('admin.graph.title')}</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {t('admin.graph.subtitle')}
        </Typography>
        {overview && (
          <Box display="flex" gap={2} mt={1}>
            <Chip label={`${overview.stats.entities.toLocaleString()} ${t('admin.graph.entities')}`} size="small" />
            <Chip label={`${overview.stats.relationships.toLocaleString()} ${t('admin.graph.relationships')}`} size="small" />
            <Chip label={`${overview.stats.communities} ${t('admin.graph.communitiesLabel')}`} size="small" />
          </Box>
        )}
      </Box>

      {/* Main content: sidebar + graph */}
      <Box sx={{ flex: 1, display: 'flex', gap: 2, minHeight: 0 }}>
        {/* Sidebar */}
        <Card sx={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
          <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2, '&:last-child': { pb: 2 } }}>
            {/* Filters */}
            <Typography variant="subtitle2" gutterBottom>
              {t('admin.graph.filters')}
            </Typography>

            {overview && (
              <>
                <Autocomplete
                  multiple
                  size="small"
                  options={overview.documents.map((d) => d.document_id)}
                  getOptionLabel={(id) =>
                    overview.documents.find((d) => d.document_id === id)?.display_name || id
                  }
                  renderOption={(props, id) => {
                    const doc = overview.documents.find((d) => d.document_id === id);
                    return (
                      <li {...props} key={id}>
                        {doc?.display_name} ({doc?.entity_count})
                      </li>
                    );
                  }}
                  value={selectedDocs}
                  onChange={(_, val) => setSelectedDocs(val)}
                  renderInput={(params) => (
                    <TextField {...params} label={t('admin.graph.filterByDocument')} />
                  )}
                  sx={{ mb: 1.5 }}
                />

                <Autocomplete
                  multiple
                  size="small"
                  options={overview.communities.map((c) => c.community_id)}
                  getOptionLabel={(id) => {
                    const c = overview.communities.find((c) => c.community_id === id);
                    return c ? `${c.title} (${c.entity_count})` : String(id);
                  }}
                  value={selectedCommunities}
                  onChange={(_, val) => setSelectedCommunities(val)}
                  renderInput={(params) => (
                    <TextField {...params} label={t('admin.graph.filterByCommunity')} />
                  )}
                  sx={{ mb: 1.5 }}
                />

                <Autocomplete
                  multiple
                  size="small"
                  options={overview.entity_types.map((t) => t.type)}
                  getOptionLabel={(type) => {
                    const et = overview.entity_types.find((t) => t.type === type);
                    return `${type} (${et?.count || 0})`;
                  }}
                  renderTags={(value, getTagProps) =>
                    value.map((type, index) => (
                      <Chip
                        {...getTagProps({ index })}
                        key={type}
                        label={type}
                        size="small"
                        sx={{ backgroundColor: getTypeColor(type), color: '#fff' }}
                      />
                    ))
                  }
                  value={selectedTypes}
                  onChange={(_, val) => setSelectedTypes(val)}
                  renderInput={(params) => (
                    <TextField {...params} label={t('admin.graph.filterByType')} />
                  )}
                  sx={{ mb: 1.5 }}
                />
              </>
            )}

            <Button
              variant="contained"
              fullWidth
              onClick={loadGraph}
              disabled={!hasFilters || graphLoading}
              startIcon={graphLoading ? <CircularProgress size={16} /> : <Download size={16} />}
              sx={{ mb: 1.5 }}
            >
              {t('admin.graph.loadGraph')}
            </Button>

            {graphData && graphData.limited && (
              <Alert severity="warning" sx={{ mb: 1.5, py: 0.5 }}>
                {t('admin.graph.limitWarning', {
                  shown: graphData.nodes.length,
                  total: graphData.total_matching,
                })}
              </Alert>
            )}

            {/* Search */}
            {graphData && graphData.nodes.length > 0 && (
              <TextField
                size="small"
                fullWidth
                placeholder={t('admin.graph.searchPlaceholder')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: <Search size={16} style={{ marginRight: 8, opacity: 0.5 }} />,
                }}
                sx={{ mb: 1.5 }}
              />
            )}

            <Divider sx={{ my: 1 }} />

            {/* Tabs: Nodes by type / Communities */}
            {graphData && graphData.nodes.length > 0 && (
              <>
                <Tabs
                  value={sidebarTab}
                  onChange={(_, v) => setSidebarTab(v)}
                  variant="fullWidth"
                  sx={{ minHeight: 36, '& .MuiTab-root': { minHeight: 36, py: 0.5, fontSize: '0.8rem' } }}
                >
                  <Tab label={t('admin.graph.tabNodes')} />
                  <Tab label={t('admin.graph.tabCommunities')} />
                </Tabs>

                <Box sx={{ flex: 1, overflow: 'auto', mt: 1 }}>
                  {sidebarTab === 0 && (
                    <List dense disablePadding>
                      {Array.from(nodesByType.entries())
                        .sort((a, b) => b[1].length - a[1].length)
                        .map(([type, nodes]) => (
                          <Box key={type}>
                            <Typography
                              variant="caption"
                              fontWeight={700}
                              sx={{ px: 1, py: 0.5, display: 'flex', alignItems: 'center', gap: 0.5 }}
                            >
                              <Box
                                sx={{
                                  width: 10,
                                  height: 10,
                                  borderRadius: '50%',
                                  backgroundColor: getTypeColor(type),
                                  flexShrink: 0,
                                }}
                              />
                              {type} ({nodes.length})
                            </Typography>
                            {nodes.slice(0, 20).map((node) => (
                              <ListItemButton
                                key={node.id}
                                dense
                                sx={{ py: 0.25, pl: 3 }}
                                onClick={() => setSearchQuery(node.name)}
                              >
                                <ListItemText
                                  primary={node.name}
                                  primaryTypographyProps={{ variant: 'body2', noWrap: true }}
                                />
                              </ListItemButton>
                            ))}
                            {nodes.length > 20 && (
                              <Typography variant="caption" color="text.secondary" sx={{ pl: 3 }}>
                                +{nodes.length - 20} more
                              </Typography>
                            )}
                          </Box>
                        ))}
                    </List>
                  )}

                  {sidebarTab === 1 && (
                    <List dense disablePadding>
                      {visibleCommunities.map((comm) => (
                        <ListItemButton
                          key={comm.community_id}
                          dense
                          onClick={() => {
                            setSelectedCommunities([comm.community_id]);
                            setSelectedDocs([]);
                            setSelectedTypes([]);
                            // Auto-load after filter change
                            setTimeout(loadGraph, 0);
                          }}
                        >
                          <ListItemText
                            primary={comm.title}
                            secondary={`${comm.entity_count} entities`}
                            primaryTypographyProps={{ variant: 'body2', noWrap: true }}
                          />
                        </ListItemButton>
                      ))}
                      {visibleCommunities.length === 0 && (
                        <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
                          {t('admin.graph.noCommunities')}
                        </Typography>
                      )}
                    </List>
                  )}
                </Box>
              </>
            )}
          </CardContent>
        </Card>

        {/* Graph Canvas */}
        <Card sx={{ flex: 1, position: 'relative', minHeight: 0 }}>
          {!graphData && !graphLoading && (
            <Box
              display="flex"
              justifyContent="center"
              alignItems="center"
              height="100%"
              flexDirection="column"
              gap={2}
            >
              <Network size={64} style={{ opacity: 0.15 }} />
              <Typography color="text.secondary">{t('admin.graph.noData')}</Typography>
            </Box>
          )}

          {graphLoading && (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <CircularProgress />
            </Box>
          )}

          {graphError && (
            <Box p={3}>
              <Alert severity="error">{graphError}</Alert>
            </Box>
          )}

          {graphData && graphData.nodes.length > 0 && (
            <SigmaContainer
              style={{ height: '100%', width: '100%' }}
              settings={{
                allowInvalidContainer: true,
                renderEdgeLabels: false,
                defaultEdgeColor: '#cbd5e1',
                defaultEdgeType: 'line',
                labelFont: 'Inter, system-ui, sans-serif',
                labelSize: 12,
                labelWeight: '500',
                enableEdgeEvents: true,
              }}
            >
              <GraphLoader graphData={graphData} onNodeMap={handleNodeMapUpdate} />
              <GraphEvents
                nodeMap={nodeMap}
                graphData={graphData}
                setTooltip={setTooltip}
              />
              <SearchHighlighter searchQuery={searchQuery} />
            </SigmaContainer>
          )}

          {graphData && graphData.nodes.length === 0 && (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography color="text.secondary">{t('admin.graph.noResults')}</Typography>
            </Box>
          )}

          <GraphTooltip tooltip={tooltip} />
        </Card>
      </Box>
    </Box>
  );
}
