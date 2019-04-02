/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.geometry.convex;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

/**
 * Computes the convex hull of a set of three dimensional points.
 *
 * <p>
 * The algorithm is a three dimensional implementation of Quickhull, as
 * described in Barber, Dobkin, and Huhdanpaa, <a
 * href=http://citeseer.ist.psu.edu/barber96quickhull.html> ``The Quickhull
 * Algorithm for Convex Hulls''</a> (ACM Transactions on Mathematical Software,
 * Vol. 22, No. 4, December 1996), and has a complexity of O(n log(n)) with
 * respect to the number of points. A well-known C implementation of Quickhull
 * that works for arbitrary dimensions is provided by <a
 * href=http://www.qhull.org>qhull</a>.
 * https://pdfs.semanticscholar.org/presentation/84bf/a054f1b059373354abfcaf9b26ee97c0839d.pdf
 * <p>
 * A hull is constructed by providing a set of points to either a constructor or
 * a {@link #build(Point3D[]) build} method. After the hull is built, its
 * vertices and faces can be retrieved using {@link #getVertices()
 * getVertices} and {@link #getHullFaces() getHullFaces}. A typical usage might
 * look like this:
 * <pre>
 *   // x y z coordinates of 6 points
 *   Point3D[] points = new Point3D[]
 *    { new Point3D (0.0,  0.0,  0.0),
 *      new Point3D (1.0,  0.5,  0.0),
 *      new Point3D (2.0,  0.0,  0.0),
 *      new Point3D (0.5,  0.5,  0.5),
 *      new Point3D (0.0,  0.0,  2.0),
 *      new Point3D (0.1,  0.2,  0.3),
 *      new Point3D (0.0,  2.0,  0.0),
 *    };
 *
 *   QuickHull3D hull = new QuickHull3D();
 *   hull.build (points);
 *
 *   System.out.println ("Vertices:");
 *   Point3D[] vertices = hull.getVertices();
 *   for (int i = 0; i < vertices.length; i++)
 *    { Point3D pnt = vertices[i];
 *      System.out.println (pnt.getX() + " " + pnt.getY() + " " + pnt.getZ());
 *    }
 *
 *   System.out.println ("HullFaces:");
 *   int[][] faceIndices = hull.getHullFaces();
 *   for (int i = 0; i < faceIndices.length; i++)
 *    { for (int k = 0; k < faceIndices[i].length; k++)
 *       { System.out.print (faceIndices[i][k] + " ");
 *       }
 *      System.out.println ("");
 *    }
 * </pre> As a convenience, there are also {@link #build(double[]) build} and
 * {@link #getVertices(double[]) getVertex} methods which pass point information
 * using an array of doubles.
 *
 * <h3><a name=distTol>Robustness</h3> Because this algorithm uses floating
 * point arithmetic, it is potentially vulnerable to errors arising from
 * numerical imprecision. We address this problem in the same way as <a
 * href=http://www.qhull.org>qhull</a>, by merging faces whose edges are not
 * clearly convex. A face is convex if its edges are convex, and an edge is
 * convex if the centroid of each adjacent plane is clearly <i>below</i> the
 * plane of the other face. The centroid is considered below a plane if its
 * distance to the plane is less than the negative of a {@link
 * #getDistanceTolerance() distance tolerance}. This tolerance represents the
 * smallest distance that can be reliably computed within the available numeric
 * precision. It is normally computed automatically from the point data,
 * although an application may {@link #setExplicitDistanceTolerance set this
 * tolerance explicitly}.
 *
 * <p>
 * Numerical problems are more likely to arise in situations where data points
 * lie on or within the faces or edges of the convex hull. We have tested
 * QuickHull3D for such situations by computing the convex hull of a random
 * point set, then adding additional randomly chosen points which lie very close
 * to the hull vertices and edges, and computing the convex hull again. The hull
 * is deemed correct if {@link #check check} returns <code>true</code>. These
 * tests have been successful for a large number of trials and so we are
 * confident that QuickHull3D is reasonably robust.
 *
 * <h3>Merged HullFaces</h3> The merging of faces means that the faces returned
 * by QuickHull3D may be convex polygons instead of triangles. If triangles are
 * desired, the application may {@link #triangulate triangulate} the faces, but
 * it should be noted that this may result in triangles which are very small or
 * thin and hence difficult to perform reliable convexity tests on. In other
 * words, triangulating a merged face is likely to restore the numerical
 * problems which the merging process removed. Hence is it possible that, after
 * triangulation, {@link #check check} will fail (the same behavior is observed
 * with triangulated output from <a
 * href=http://www.qhull.org>qhull</a>).
 *
 * <h3>Degenerate Input</h3>It is assumed that the input points are
 * non-degenerate in that they are not coincident, colinear, or colplanar, and
 * thus the convex hull has a non-zero volume. If the input points are detected
 * to be degenerate within the
 * {@link #getDistanceTolerance() distance tolerance}, an
 * IllegalArgumentException will be thrown.
 *
 * @author John E. Lloyd, Fall 2004
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class QuickHull3D {

    /**
     * Specifies that the distance tolerance should be computed automatically
     * from the input point data.
     */
    private static final double AUTOMATIC_TOLERANCE = -1;

    // ========================================================
    // estimated size of the point set
    private double charLength;

    // ========================================================
    private Vertex[] pointBuffer = new Vertex[0];
    private final HullFace[] discardedHullFaces = new HullFace[3];
    private int[] vertexPointIndices = new int[0];

    private final Vertex[] maxVtxs = new Vertex[3];
    private final Vertex[] minVtxs = new Vertex[3];

    private final List<HullFace> faces = new ArrayList<>(16);
    private final List<HalfEdge> horizon = new ArrayList<>(16);

    private final HullFaceList newHullFaces = new HullFaceList();
    private final VertexList unclaimed = new VertexList();
    private final VertexList claimed = new VertexList();

    private int numVertices;
    private int numPoints;

    private enum NONCONVEX_Type {
        NONCONVEX_WRT_LARGER_FACE, NONCONVEX
    };

    private double explicitTolerance = AUTOMATIC_TOLERANCE;
    private double tolerance;

    /**
     * Creates an empty convex hull object.
     */
    public QuickHull3D() {
    }

    /**
     * Creates a convex hull object and initializes it to the convex hull of a
     * set of points whose coordinates are given by an array of doubles.
     *
     * @param coords x, y, and z coordinates of each input point. The length of
     * this array will be three times the the number of input points.
     * @throws IllegalArgumentException the number of input points is less than
     * four, or the points appear to be coincident, colinear, or coplanar.
     */
    public QuickHull3D(double[] coords)
            throws IllegalArgumentException {
        build(coords, coords.length / 3);
    }

    /**
     * Creates a convex hull object and initializes it to the convex hull of a
     * set of points.
     *
     * @param points input points.
     * @throws IllegalArgumentException the number of input points is less than
     * four, or the points appear to be coincident, colinear, or coplanar.
     */
    public QuickHull3D(Point3D[] points)
            throws IllegalArgumentException {
        build(points, points.length);
    }

    // <editor-fold defaultstate="collapsed" desc="Initialization">
    private void initBuffers(int nump) {
        if (pointBuffer.length < nump) {
            Vertex[] newBuffer = new Vertex[nump];
            vertexPointIndices = new int[nump];
            System.arraycopy(pointBuffer, 0, newBuffer,
                    0, pointBuffer.length);

            for (int i = pointBuffer.length; i < nump; i++) {
                newBuffer[i] = new Vertex();
            }
            pointBuffer = newBuffer;
        }
        faces.clear();
        claimed.clear();
        numPoints = nump;
    }

    private void computeMaxAndMin() {

        for (int i = 0; i < 3; i++) {
            maxVtxs[i] = minVtxs[i] = pointBuffer[0];
        }
        Vector3D max = new Vector3D((pointBuffer[0].toArray()));
        Vector3D min = new Vector3D((pointBuffer[0].toArray()));

        for (int i = 1; i < numPoints; i++) {
            Point3D pnt = pointBuffer[i];
            if (pnt.getX() > max.getX()) {
                max = new Vector3D(pnt.getX(), max.getY(), max.getZ());
                maxVtxs[0] = pointBuffer[i];
            } else if (pnt.getX() < min.getX()) {
                min = new Vector3D(pnt.getX(), min.getY(), min.getZ());
                minVtxs[0] = pointBuffer[i];
            }
            if (pnt.getY() > max.getY()) {
                max = new Vector3D(max.getX(), pnt.getY(), max.getZ());
                maxVtxs[1] = pointBuffer[i];
            } else if (pnt.getY() < min.getY()) {
                min = new Vector3D(min.getX(), pnt.getY(), min.getZ());
                minVtxs[1] = pointBuffer[i];
            }
            if (pnt.getZ() > max.getZ()) {
                max = new Vector3D(max.getX(), max.getY(), pnt.getZ());
                maxVtxs[2] = pointBuffer[i];
            } else if (pnt.getZ() < min.getZ()) {
                min = new Vector3D(min.getX(), min.getY(), pnt.getZ());
                minVtxs[2] = pointBuffer[i];
            }
        }

        // this epsilon formula comes from QuickHull, and I'm
        // not about to quibble.
        charLength = Math.max(max.getX() - min.getX(), max.getY() - min.getY());
        charLength = Math.max(max.getZ() - min.getZ(), charLength);
        if (explicitTolerance == AUTOMATIC_TOLERANCE) {
            tolerance = 3 * Math.ulp(1.0)
                    * (Math.max(Math.abs(max.getX()), Math.abs(min.getX()))
                    + Math.max(Math.abs(max.getY()), Math.abs(min.getY()))
                    + Math.max(Math.abs(max.getZ()), Math.abs(min.getZ())));
        } else {
            tolerance = explicitTolerance;
        }
    }

    /**
     * Creates the initial simplex from which the hull will be built.
     */
    private void createInitialSimplex()
            throws IllegalArgumentException {
        double max = 0;
        int imax = 0;

        for (int i = 0; i < 3; i++) {
            double diff;
            switch (i) {
                case 0:
                    diff = maxVtxs[i].getX() - minVtxs[i].getX();
                    break;
                case 1:
                    diff = maxVtxs[i].getY() - minVtxs[i].getY();
                    break;
                default:
                    diff = maxVtxs[i].getZ() - minVtxs[i].getZ();
                    break;
            }
            if (diff > max) {
                max = diff;
                imax = i;
            }
        }

        if (max <= tolerance) {
            throw new IllegalArgumentException(
                    "Input points appear to be coincident");
        }
        Vertex[] vtx = new Vertex[4];
        // set first two vertices to be those with the greatest
        // one dimensional separation

        vtx[0] = maxVtxs[imax];
        vtx[1] = minVtxs[imax];

        // set third vertex to be the vertex farthest from
        // the line between vtx0 and vtx1
        Vector3D diff02, xprod;
        Vector3D nrml = new Vector3D(0.0, 0.0, 0.0);
        double maxSqr = 0;
        Vector3D u01 = (vtx[1].subtract(vtx[0])).normalize();
        for (int i = 0; i < numPoints; i++) {
            diff02 = pointBuffer[i].subtract(vtx[0]);
            xprod = u01.crossProduct(diff02);
            double lenSqr = xprod.getNormSq();
            if (lenSqr > maxSqr
                    && pointBuffer[i] != vtx[0]
                    && // paranoid
                    pointBuffer[i] != vtx[1]) {
                maxSqr = lenSqr;
                vtx[2] = pointBuffer[i];
                nrml = xprod;
            }
        }
        if (Math.sqrt(maxSqr) <= 100 * tolerance) {
            throw new IllegalArgumentException(
                    "Input points appear to be colinear");
        }
        nrml = nrml.normalize();

        // recompute nrml to make sure it is normal to u10 - otherwise could
        // be errors in case vtx[2] is close to u10
        Vector3D res = u01.scalarMultiply(nrml.dotProduct(u01)); // component of nrml along u01
        nrml = (nrml.subtract(res)).normalize();

        double maxDist = 0;
        double d0 = vtx[2].dotProduct(nrml);
        for (int i = 0; i < numPoints; i++) {
            double dist = Math.abs(pointBuffer[i].dotProduct(nrml) - d0);
            if (dist > maxDist
                    && pointBuffer[i] != vtx[0]
                    && // paranoid
                    pointBuffer[i] != vtx[1]
                    && pointBuffer[i] != vtx[2]) {
                maxDist = dist;
                vtx[3] = pointBuffer[i];
            }
        }
        if (Math.abs(maxDist) <= 100 * tolerance) {
            throw new IllegalArgumentException(
                    "Input points appear to be coplanar");
        }

        HullFace[] tris = new HullFace[4];

        if (vtx[3].dotProduct(nrml) - d0 < 0) {
            tris[0] = HullFace.createTriangle(vtx[0], vtx[1], vtx[2]);
            tris[1] = HullFace.createTriangle(vtx[3], vtx[1], vtx[0]);
            tris[2] = HullFace.createTriangle(vtx[3], vtx[2], vtx[1]);
            tris[3] = HullFace.createTriangle(vtx[3], vtx[0], vtx[2]);

            for (int i = 0; i < 3; i++) {
                int k = (i + 1) % 3;
                tris[i + 1].getEdge(1).setOpposite(tris[k + 1].getEdge(0));
                tris[i + 1].getEdge(2).setOpposite(tris[0].getEdge(k));
            }
        } else {
            tris[0] = HullFace.createTriangle(vtx[0], vtx[2], vtx[1]);
            tris[1] = HullFace.createTriangle(vtx[3], vtx[0], vtx[1]);
            tris[2] = HullFace.createTriangle(vtx[3], vtx[1], vtx[2]);
            tris[3] = HullFace.createTriangle(vtx[3], vtx[2], vtx[0]);

            for (int i = 0; i < 3; i++) {
                int k = (i + 1) % 3;
                tris[i + 1].getEdge(0).setOpposite(tris[k + 1].getEdge(1));
                tris[i + 1].getEdge(2).setOpposite(tris[0].getEdge((3 - i) % 3));
            }
        }

        for (int i = 0; i < 4; i++) {
            faces.add(tris[i]);
        }

        for (int i = 0; i < numPoints; i++) {
            Vertex v = pointBuffer[i];

            if (v == vtx[0] || v == vtx[1]
                    || v == vtx[2] || v == vtx[3]) {
                continue;
            }

            maxDist = tolerance;
            HullFace maxHullFace = null;
            for (int k = 0; k < 4; k++) {
                double dist = tris[k].distanceToPlane(v);
                if (dist > maxDist) {
                    maxHullFace = tris[k];
                    maxDist = dist;
                }
            }
            if (maxHullFace != null) {
                addPointToHullFace(v, maxHullFace);
            }
        }
    }

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Build">
    /**
     * Constructs the convex hull of a set of points whose coordinates are given
     * by an array of doubles.
     *
     * @param coords x, y, and z coordinates of each input point. The length of
     * this array will be three times the number of input points.
     * @throws IllegalArgumentException the number of input points is less than
     * four, or the points appear to be coincident, colinear, or coplanar.
     */
    public void build(double[] coords)
            throws IllegalArgumentException {
        build(coords, coords.length / 3);
    }

    /**
     * Constructs the convex hull of a set of points whose coordinates are given
     * by an array of doubles.
     *
     * @param coords x, y, and z coordinates of each input point. The length of
     * this array must be at least three times <code>nump</code>.
     * @param nump number of input points
     * @throws IllegalArgumentException the number of input points is less than
     * four or greater than 1/3 the length of <code>coords</code>, or the points
     * appear to be coincident, colinear, or coplanar.
     */
    public final void build(double[] coords, int nump)
            throws IllegalArgumentException {
        if (nump < 4) {
            throw new IllegalArgumentException(
                    "Less than four input points specified");
        }
        if (coords.length / 3 < nump) {
            throw new IllegalArgumentException(
                    "Coordinate array too small for specified number of points");
        }
        initBuffers(nump);
        setPoints(coords, nump);
        buildHull();
    }

    /**
     * Constructs the convex hull of a set of points.
     *
     * @param points input points
     * @throws IllegalArgumentException the number of input points is less than
     * four, or the points appear to be coincident, colinear, or coplanar.
     */
    public void build(Point3D[] points)
            throws IllegalArgumentException {
        build(points, points.length);
    }

    /**
     * Constructs the convex hull of a set of points.
     *
     * @param points input points
     * @param nump number of input points
     * @throws IllegalArgumentException the number of input points is less than
     * four or greater then the length of <code>points</code>, or the points
     * appear to be coincident, colinear, or coplanar.
     */
    public final void build(Point3D[] points, int nump)
            throws IllegalArgumentException {
        if (nump < 4) {
            throw new IllegalArgumentException(
                    "Less than four input points specified");
        }
        if (points.length < nump) {
            throw new IllegalArgumentException(
                    "Point array too small for specified number of points");
        }
        initBuffers(nump);
        setPoints(points, nump);
        buildHull();
    }

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Internal Getters">
    /**
     * Returns the vertex points in this hull.
     *
     * @return array of vertex points
     * @see QuickHull3D#getVertices(double[])
     * @see QuickHull3D#getHullFaces()
     */
    public Point3D[] getVertices() {
        Point3D[] vtxs = new Point3D[numVertices];
        for (int i = 0; i < numVertices; i++) {
            vtxs[i] = pointBuffer[vertexPointIndices[i]];
        }
        return vtxs;
    }

    /**
     * Returns the coordinates of the vertex points of this hull.
     *
     * @param coords returns the x, y, z coordinates of each vertex. This length
     * of this array must be at least three times the number of vertices.
     * @return the number of vertices
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getHullFaces()
     */
    public int getVertices(double[] coords) {
        for (int i = 0; i < numVertices; i++) {
            Point3D pnt = pointBuffer[vertexPointIndices[i]];
            coords[i * 3 + 0] = pnt.getX();
            coords[i * 3 + 1] = pnt.getY();
            coords[i * 3 + 2] = pnt.getZ();
        }
        return numVertices;
    }

    /**
     * Returns an array specifing the index of each hull vertex with respect to
     * the original input points.
     *
     * @return vertex indices with respect to the original points
     */
    public int[] getVertexPointIndices() {
        int[] indices = new int[numVertices];
        System.arraycopy(vertexPointIndices, 0,
                indices, 0, numVertices);
        return indices;
    }

    /**
     * Returns the number of faces in this hull.
     *
     * @return number of faces
     */
    public int getNumHullFaces() {
        return faces.size();
    }

    /**
     * Returns the faces associated with this hull.
     *
     * <p>
     * Each face is represented by an integer array which gives the indices of
     * the vertices. These indices are numbered relative to the hull vertices,
     * are zero-based, and are arranged counter-clockwise. More control over the
     * index format can be obtained using
     * {@link #getHullFaces(int) getHullFaces(indexFlags)}.
     *
     * @return array of integer arrays, giving the vertex indices for each face.
     * @see QuickHull3D#getVertices()
     * @see QuickHull3D#getHullFaces(int)
     */
    public int[][] getHullFaces() {
        return getHullFaces(0);
    }

    /**
     * Returns the faces associated with this hull.
     *
     * <p>
     * Each face is represented by an integer array which gives the indices of
     * the vertices. By default, these indices are numbered with respect to the
     * hull vertices (as opposed to the input points), are zero-based, and are
     * arranged counter-clockwise. However, this can be changed by setting {@link #POINT_RELATIVE
     * POINT_RELATIVE}, {@link #INDEXED_FROM_ONE INDEXED_FROM_ONE}, or
     * {@link #CLOCKWISE CLOCKWISE} in the indexFlags parameter.
     *
     * @param indexFlags specifies index characteristics (0 results in the
     * default)
     * @return array of integer arrays, giving the vertex indices for each face.
     * @see QuickHull3D#getVertices()
     */
    public int[][] getHullFaces(int indexFlags) {
        int[][] allHullFaces = new int[faces.size()][];
        int k = 0;
        for (HullFace face : faces) {
            allHullFaces[k] = new int[face.numVertices()];
            getHullFaceIndices(allHullFaces[k], face, indexFlags);
            k++;
        }
        return allHullFaces;
    }

    private void getHullFaceIndices(int[] indices, HullFace face, int flags) {
        boolean ccw = ((flags & 0x1) == 0);
        boolean indexedFromOne = ((flags & 0x2) != 0);
        boolean pointRelative = ((flags & 0x8) != 0);

        HalfEdge hedge = face.getFirstEdge();
        int k = 0;
        do {
            int idx = hedge.getHead().index;
            if (pointRelative) {
                idx = vertexPointIndices[idx];
            }
            if (indexedFromOne) {
                idx++;
            }
            indices[k++] = idx;
            hedge = (ccw ? hedge.next : hedge.prev);
        } while (hedge != face.getFirstEdge());
    }

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Hull Face Manipulation">
    private void addPointToHullFace(Vertex vtx, HullFace face) {
        vtx.face = face;

        if (face.getOutside() == null) {
            claimed.add(vtx);
        } else {
            claimed.insertBefore(vtx, face.getOutside());
        }
        face.setOutside(vtx);
    }

    private void removePointFromHullFace(Vertex vtx, HullFace face) {
        if (vtx == face.getOutside()) {
            if (vtx.next != null && vtx.next.face == face) {
                face.setOutside(vtx.next);
            } else {
                face.setOutside(null);
            }
        }
        claimed.delete(vtx);
    }

    private Vertex removeAllPointsFromHullFace(HullFace face) {
        if (face.getOutside() != null) {
            Vertex end = face.getOutside();
            while (end.next != null && end.next.face == face) {
                end = end.next;
            }
            claimed.delete(face.getOutside(), end);
            end.next = null;
            return face.getOutside();
        } else {
            return null;
        }
    }

    /**
     * Triangulates any non-triangular hull faces. In some cases, due to
     * precision issues, the resulting triangles may be very thin or small, and
     * hence appear to be non-convex (this same limitation is present in
     * <a href=http://www.qhull.org>qhull</a>).
     */
    public void triangulate() {
        double minArea = 1000 * charLength * Math.ulp(1.0);
        newHullFaces.clear();
        faces.stream().filter((face)
                -> (face.getMark() == HullFace.STATUS.VISIBLE))
                .forEachOrdered((face) -> {
                    face.triangulate(newHullFaces, minArea);
                    // splitHullFace (face);
                });

        for (HullFace face = newHullFaces.first(); face != null;
                face = face.getNext()) {
            faces.add(face);
        }
    }

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Internal Hull Computations">
    private void resolveUnclaimedPoints(HullFaceList newHullFaces) {
        Vertex vtxNext = unclaimed.first();
        for (Vertex vtx = vtxNext; vtx != null; vtx = vtxNext) {
            vtxNext = vtx.next;

            double maxDist = tolerance;
            HullFace maxHullFace = null;
            for (HullFace newHullFace = newHullFaces.first(); newHullFace != null;
                    newHullFace = newHullFace.getNext()) {
                if (newHullFace.getMark() == HullFace.STATUS.VISIBLE) {
                    double dist = newHullFace.distanceToPlane(vtx);
                    if (dist > maxDist) {
                        maxDist = dist;
                        maxHullFace = newHullFace;
                    }
                    if (maxDist > 1000 * tolerance) {
                        break;
                    }
                }
            }
            if (maxHullFace != null) {
                addPointToHullFace(vtx, maxHullFace);
            }
        }
    }

    private void deleteHullFacePoints(HullFace face, HullFace absorbingHullFace) {
        Vertex faceVtxs = removeAllPointsFromHullFace(face);
        if (faceVtxs != null) {
            if (absorbingHullFace == null) {
                unclaimed.addAll(faceVtxs);
            } else {
                Vertex vtxNext = faceVtxs;
                for (Vertex vtx = vtxNext; vtx != null; vtx = vtxNext) {
                    vtxNext = vtx.next;
                    double dist = absorbingHullFace.distanceToPlane(vtx);
                    if (dist > tolerance) {
                        addPointToHullFace(vtx, absorbingHullFace);
                    } else {
                        unclaimed.add(vtx);
                    }
                }
            }
        }
    }

    private double oppHullFaceDistance(HalfEdge he) {
        return he.face.distanceToPlane(he.opposite.face.getCentroid());
    }

    private boolean doAdjacentMerge(HullFace face, NONCONVEX_Type mergeType) {
        HalfEdge hedge = face.getFirstEdge();

        boolean convex = true;
        do {
            HullFace oppHullFace = hedge.getOppositeHullFace();
            boolean merge = false;

            if (mergeType == NONCONVEX_Type.NONCONVEX) { // then merge faces if they are definitively non-convex
                if (oppHullFaceDistance(hedge) > -tolerance
                        || oppHullFaceDistance(hedge.opposite) > -tolerance) {
                    merge = true;
                }
            } else { // merge faces if they are parallel or non-convex
                // wrt to the larger face; otherwise, just mark
                // the face non-convex for the second pass.
                if (face.getArea() > oppHullFace.getArea()) {
                    if (oppHullFaceDistance(hedge) > -tolerance) {
                        merge = true;
                    } else if (oppHullFaceDistance(hedge.opposite) > -tolerance) {
                        convex = false;
                    }
                } else {
                    if (oppHullFaceDistance(hedge.opposite) > -tolerance) {
                        merge = true;
                    } else if (oppHullFaceDistance(hedge) > -tolerance) {
                        convex = false;
                    }
                }
            }

            if (merge) {
                int numd = face.mergeAdjacentHullFace(hedge, discardedHullFaces);
                for (int idx = 0; idx < numd; idx++) {
                    deleteHullFacePoints(discardedHullFaces[idx], face);
                }
                return true;
            }
            hedge = hedge.next;
        } while (hedge != face.getFirstEdge());

        if (!convex) {
            face.setMark(HullFace.STATUS.NON_CONVEX);
        }
        return false;
    }

    private void calculateHorizon(Point3D eyePnt, HalfEdge edge0, HullFace face) {

        deleteHullFacePoints(face, null);
        face.setMark(HullFace.STATUS.DELETED);

        HalfEdge edge;
        if (edge0 == null) {
            edge0 = face.getEdge(0);
            edge = edge0;
        } else {
            edge = edge0.getNext();
        }

        do {
            HullFace oppHullFace = edge.getOppositeHullFace();
            if (oppHullFace.getMark() == HullFace.STATUS.VISIBLE) {
                if (oppHullFace.distanceToPlane(eyePnt) > tolerance) {
                    calculateHorizon(eyePnt, edge.getOpposite(),
                            oppHullFace);
                } else {
                    horizon.add(edge);
                }
            }
            edge = edge.getNext();
        } while (edge != edge0);
    }

    private HalfEdge addAdjoiningHullFace(Vertex eyeVtx, HalfEdge he) {
        HullFace face = HullFace.createTriangle(
                eyeVtx, he.getTail(), he.getHead());
        faces.add(face);
        face.getEdge(-1).setOpposite(he.getOpposite());
        return face.getEdge(0);
    }

    private void addNewHullFaces(HullFaceList newHullFaces, Vertex eyeVtx) {
        newHullFaces.clear();

        HalfEdge hedgeSidePrev = new HalfEdge();
        HalfEdge hedgeSideBegin = new HalfEdge();

        for (HalfEdge horizonHe : horizon) {
            HalfEdge hedgeSide = addAdjoiningHullFace(eyeVtx, horizonHe);

            if (!hedgeSidePrev.isEmpty()) {
                hedgeSide.next.setOpposite(hedgeSidePrev);
            } else {
                hedgeSideBegin = hedgeSide;
            }
            newHullFaces.add(hedgeSide.getHullFace());
            hedgeSidePrev = hedgeSide;
        }
        hedgeSideBegin.next.setOpposite(hedgeSidePrev);
    }

    private Vertex nextPointToAdd() {
        if (!claimed.isEmpty()) {
            HullFace eyeHullFace = claimed.first().face;
            Vertex eyeVtx = null;
            double maxDist = 0;

            for (Vertex vtx = eyeHullFace.getOutside();
                    vtx != null && vtx.face == eyeHullFace;
                    vtx = vtx.next) {
                double dist = eyeHullFace.distanceToPlane(vtx);
                if (dist > maxDist) {
                    maxDist = dist;
                    eyeVtx = vtx;
                }
            }
            return eyeVtx;
        } else {
            return null;
        }
    }

    private void addPointToHull(Vertex eyeVtx) {
        horizon.clear();
        unclaimed.clear();

        removePointFromHullFace(eyeVtx, eyeVtx.face);
        calculateHorizon(eyeVtx, null, eyeVtx.face);

        newHullFaces.clear();
        addNewHullFaces(newHullFaces, eyeVtx);

        // first merge pass ... merge faces which are non-convex
        // as determined by the larger face
        for (HullFace face = newHullFaces.first(); face != null; face = face.getNext()) {
            if (face.getMark() == HullFace.STATUS.VISIBLE) {
                boolean allDone = Boolean.TRUE;
                while (allDone) {
                    allDone = doAdjacentMerge(face, NONCONVEX_Type.NONCONVEX_WRT_LARGER_FACE);
                }

            }
        }

        // second merge pass ... merge faces which are non-convex
        // wrt either face      
        for (HullFace face = newHullFaces.first(); face != null; face = face.getNext()) {
            if (face.getMark() == HullFace.STATUS.NON_CONVEX) {
                face.setMark(HullFace.STATUS.VISIBLE);

                boolean allDone = Boolean.TRUE;
                while (allDone) {
                    allDone = doAdjacentMerge(face, NONCONVEX_Type.NONCONVEX);
                }
            }
        }
        resolveUnclaimedPoints(newHullFaces);
    }

    private void buildHull() {
        Vertex eyeVtx;

        computeMaxAndMin();
        createInitialSimplex();
        while ((eyeVtx = nextPointToAdd()) != null) {
            addPointToHull(eyeVtx);
        }
        reindexHullFacesAndVertices();
    }

    private void markHullFaceVertices(HullFace face, int mark) {
        HalfEdge he0 = face.getFirstEdge();
        HalfEdge he = he0;
        do {
            he.getHead().index = mark;
            he = he.next;
        } while (he != he0);
    }

    private void reindexHullFacesAndVertices() {
        for (int i = 0; i < numPoints; i++) {
            pointBuffer[i].index = -1;
        }

        // remove inactive faces and mark active vertices
        for (Iterator it = faces.iterator(); it.hasNext();) {
            HullFace face = (HullFace) it.next();
            if (face.getMark() != HullFace.STATUS.VISIBLE) {
                it.remove();
            } else {
                markHullFaceVertices(face, 0);
            }
        }

        // reindex vertices
        numVertices = 0;
        for (int i = 0; i < numPoints; i++) {
            Vertex vtx = pointBuffer[i];
            if (vtx.index == 0) {
                vertexPointIndices[numVertices] = i;
                vtx.index = numVertices++;
            }
        }
    }

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Getter and Setter">
    /**
     * Returns the distance tolerance that was used for the most recently
     * computed hull. The distance tolerance is used to determine when faces are
     * unambiguously convex with respect to each other, and when points are
     * unambiguously above or below a face plane, in the presence of
     * <a href=#distTol>numerical imprecision</a>. Normally, this tolerance is
     * computed automatically for each set of input points, but it can be set
     * explicitly by the application.
     *
     * @return distance tolerance
     * @see QuickHull3D#setExplicitDistanceTolerance
     */
    public double getDistanceTolerance() {
        return tolerance;
    }

    /**
     * Sets an explicit distance tolerance for convexity tests. If
     * {@link #AUTOMATIC_TOLERANCE AUTOMATIC_TOLERANCE} is specified (the
     * default), then the tolerance will be computed automatically from the
     * point data.
     *
     * @param tol explicit tolerance
     * @see #getDistanceTolerance
     */
    public void setExplicitDistanceTolerance(double tol) {
        explicitTolerance = tol;
    }

    /**
     * Returns the explicit distance tolerance.
     *
     * @return explicit tolerance
     * @see #setExplicitDistanceTolerance
     */
    public double getExplicitDistanceTolerance() {
        return explicitTolerance;
    }

    /**
     * Returns the number of vertices in this hull.
     *
     * @return number of vertices
     */
    public int getNumVertices() {
        return numVertices;
    }

    private void setPoints(double[] coords, int nump) {
        for (int i = 0; i < nump; i++) {
            pointBuffer[i] = new Vertex(
                    coords[i * 3 + 0], coords[i * 3 + 1], coords[i * 3 + 2], i);
        }
    }

    private void setPoints(Point3D[] pnts, int nump) {
        for (int i = 0; i < nump; i++) {
            pointBuffer[i] = new Vertex(
                    pnts[i], i);
        }
    }

    // </editor-fold>
}
