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

import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

/**
 * Basic triangular face used to form the hull.
 *
 * <p>
 * The information stored for each face consists of a planar normal, a planar
 * offset, and a doubly-linked list of three <a
 * href=HalfEdge>HalfEdges</a> which surround the face in a counter-clockwise
 * direction.
 *
 * @author John E. Lloyd,
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class HullFace {

    // =============================================================
    private Vertex outside;
    private HalfEdge he0;
    private Vector3D normal;
    private Point3D centroid;

    // =============================================================
    private double area;
    private double planeOffset;
    private int numVerts;

    private HullFace next;

    // =============================================================
    public enum STATUS {
        VISIBLE, NON_CONVEX, DELETED
    };

    private STATUS mark = STATUS.VISIBLE;

    // =============================================================
    /**
     *
     */
    public HullFace() {
        normal = new Vector3D(0.0, 0.0, 0.0);
        centroid = new Point3D();
        mark = STATUS.VISIBLE;
    }

    /**
     *
     * @param vtxArray The set of vertices
     * @param indices The set of indices
     * @return The face
     */
    public static HullFace create(Vertex[] vtxArray, int[] indices) {
        HullFace face = new HullFace();
        HalfEdge hePrev = new HalfEdge();

        for (int idx = 0; idx < indices.length; idx++) {
            HalfEdge he = new HalfEdge(vtxArray[indices[idx]], face);
            if (!hePrev.isEmpty()) {
                he.setPrev(hePrev);
                hePrev.setNext(he);
            } else {
                face.he0 = he;
            }
            hePrev = he;
        }

        face.he0.setPrev(hePrev);
        hePrev.setNext(face.he0);

        // compute the normal and offset
        face.computeNormalAndCentroid();
        return face;
    }

    // =============================================================
    // <editor-fold defaultstate="collapsed" desc="Generate centroid and normal">
    private void computeCentroid() {
        centroid = new Point3D(0.0, 0.0, 0.0);
        HalfEdge he = he0;

        do {
            centroid = new Point3D(centroid.add(he.getHead()));
            he = he.next;
        } while (he != he0);

        centroid = new Point3D(
                centroid.scalarMultiply(1 / (double) numVerts));
    }

    private void computeNormal(double minArea) {
        computeNormal();

        if (area < minArea) {
            // make the normal more robust by removing
            // components parallel to the longest edge

            HalfEdge hedgeMax = new HalfEdge();
            double lenSqrMax = 0;
            HalfEdge hedge = he0;

            do {
                double lenSqr = hedge.lengthSquared();
                if (lenSqr > lenSqrMax) {
                    hedgeMax = hedge;
                    lenSqrMax = lenSqr;
                }
                hedge = hedge.next;
            } while (hedge != he0);

            Point3D p2 = hedgeMax.getHead();
            Point3D p1 = hedgeMax.getTail();

            double lenMax = Math.sqrt(lenSqrMax);

            Vector3D uVector = (p2.subtract(p1)).scalarMultiply(1.0 / lenMax);

            double dot = normal.dotProduct(uVector);

            normal = (normal.subtract((uVector)
                    .scalarMultiply(dot))).normalize();
        }
    }

    private void computeNormal() {
        HalfEdge he1 = he0.next;
        HalfEdge he2 = he1.next;

        Point3D p0 = he0.getHead();
        Point3D p2 = he1.getHead();

        Vector3D d2 = p2.subtract(p0);

        normal = new Vector3D(0.0, 0.0, 0.0);

        numVerts = 2;

        while (he2 != he0) {

            Vector3D d1 = new Vector3D(d2.toArray());
            p2 = he2.getHead();
            d2 = p2.subtract(p0);

            normal = normal.add(
                    (new Vector3D(d1.crossProduct(d2).toArray())));

            he2 = he2.next;
            numVerts++;
        }

        area = normal.getNorm();
        normal = normal.scalarMultiply(1 / area);
    }

    private void computeNormalAndCentroid() {
        computeNormal();
        computeCentroid();

        planeOffset = normal.dotProduct(centroid);
        int numv = 0;
        HalfEdge he = he0;

        do {
            numv++;
            he = he.next;
        } while (he != he0);

        if (numv != numVerts) {
            throw new ArithmeticException(
                    "face " + getVertexString()
                    + " numVerts=" + numVerts + " should be " + numv);
        }
    }

    private void computeNormalAndCentroid(double minArea) {
        computeNormal(minArea);
        computeCentroid();
        planeOffset = normal.dotProduct(centroid);
    }
    // </editor-fold>

    // =============================================================
    // <editor-fold defaultstate="collapsed" desc="Triangle and Faces">
    /**
     *
     * @param v0
     * @param v1
     * @param v2
     * @return
     */
    public static HullFace createTriangle(Vertex v0, Vertex v1, Vertex v2) {
        return createTriangle(v0, v1, v2, 0);
    }

    /**
     * Constructs a triangle Face from vertices v0, v1, and v2.
     *
     * @param v0 first vertex
     * @param v1 second vertex
     * @param v2 third vertex
     * @param minArea
     * @return
     */
    public static HullFace createTriangle(Vertex v0, Vertex v1, Vertex v2,
            double minArea) {
        HullFace face = new HullFace();
        HalfEdge he0 = new HalfEdge(v0, face);
        HalfEdge he1 = new HalfEdge(v1, face);
        HalfEdge he2 = new HalfEdge(v2, face);

        he0.prev = he2;
        he0.next = he1;
        he1.prev = he0;
        he1.next = he2;
        he2.prev = he1;
        he2.next = he0;

        face.he0 = he0;

        // compute the normal and offset
        face.computeNormalAndCentroid(minArea);
        return face;
    }

    /**
     *
     * @param hedgeAdj
     * @param discarded
     * @return
     */
    public int mergeAdjacentHullFace(HalfEdge hedgeAdj, HullFace[] discarded) {
        HullFace oppFace = hedgeAdj.getOppositeHullFace();
        int numDiscarded = 0;

        discarded[numDiscarded++] = oppFace;
        oppFace.mark = STATUS.DELETED;

        HalfEdge hedgeOpp = hedgeAdj.getOpposite();

        HalfEdge hedgeAdjPrev = hedgeAdj.prev;
        HalfEdge hedgeAdjNext = hedgeAdj.next;
        HalfEdge hedgeOppPrev = hedgeOpp.prev;
        HalfEdge hedgeOppNext = hedgeOpp.next;

        while (hedgeAdjPrev.getOppositeHullFace() == oppFace) {
            hedgeAdjPrev = hedgeAdjPrev.prev;
            hedgeOppNext = hedgeOppNext.next;
        }

        while (hedgeAdjNext.getOppositeHullFace() == oppFace) {
            hedgeOppPrev = hedgeOppPrev.prev;
            hedgeAdjNext = hedgeAdjNext.next;
        }

        HalfEdge hedge;

        for (hedge = hedgeOppNext; hedge != hedgeOppPrev.next;
                hedge = hedge.next) {
            hedge.face = this;
        }

        if (hedgeAdj == he0) {
            he0 = hedgeAdjNext;
        }

        // handle the half edges at the head
        HullFace discardedFace;
        discardedFace = connectHalfEdges(hedgeOppPrev, hedgeAdjNext);
        if (discardedFace != null) {
            discarded[numDiscarded++] = discardedFace;
        }

        // handle the half edges at the tail
        discardedFace = connectHalfEdges(hedgeAdjPrev, hedgeOppNext);
        if (discardedFace != null) {
            discarded[numDiscarded++] = discardedFace;
        }

        computeNormalAndCentroid();
        checkConsistency();

        return numDiscarded;
    }

    /**
     *
     * @param newFaces
     * @param minArea
     */
    public void triangulate(HullFaceList newFaces, double minArea) {
        HalfEdge hedge;

        if (numVertices() < 4) {
            return;
        }

        Vertex v0 = he0.getHead();

        hedge = he0.next;
        HalfEdge oppPrev = hedge.opposite;
        HullFace face0 = null;

        for (hedge = hedge.next; hedge != he0.prev; hedge = hedge.next) {
            HullFace face = createTriangle(v0, hedge.prev.getHead(),
                    hedge.getHead(), minArea);
            face.he0.next.setOpposite(oppPrev);
            face.he0.prev.setOpposite(hedge.opposite);
            oppPrev = face.he0;

            newFaces.add(face);
            if (face0 == null) {
                face0 = face;
            }
        }

        hedge = new HalfEdge(he0.prev.prev.getHead(), this);
        hedge.setOpposite(oppPrev);

        hedge.prev = he0;
        hedge.prev.next = hedge;

        hedge.next = he0.prev;
        hedge.next.prev = hedge;

        computeNormalAndCentroid(minArea);
        checkConsistency();

        for (HullFace face = face0; face != null; face = face.next) {
            face.checkConsistency();
        }

    }

    // </editor-fold>
    
    // =============================================================
    // <editor-fold defaultstate="collapsed" desc="Edge">
    /**
     * Finds the half-edge within this face which has tail <code>vt</code> and
     * head <code>vh</code>.
     *
     * @param vt tail point
     * @param vh head point
     * @return the half-edge, or null if none is found.
     */
    public HalfEdge findEdge(Vertex vt, Vertex vh) {
        HalfEdge he = he0;

        do {
            if (he.getHead() == vh && he.getTail() == vt) {
                return he;
            }
            he = he.next;
        } while (he != he0);

        return null;
    }

    private HullFace connectHalfEdges(HalfEdge hedgePrev, HalfEdge hedge) {
        HullFace discardedFace = new HullFace();

        if (hedgePrev.getOppositeHullFace() == hedge.getOppositeHullFace()) {
            // then there is a redundant edge that we can get rid off

            HullFace oppFace = hedge.getOppositeHullFace();
            HalfEdge hedgeOpp;

            if (hedgePrev == he0) {
                he0 = hedge;
            }
            // then we can get rid of the opposite face altogether
            if (oppFace.numVertices() == 3) {
                hedgeOpp = hedge.getOpposite().prev.getOpposite();

                oppFace.mark = STATUS.DELETED;
                discardedFace = oppFace;
            } else {
                hedgeOpp = hedge.getOpposite().next;

                if (oppFace.he0 == hedgeOpp.prev) {
                    oppFace.he0 = hedgeOpp;
                }
                hedgeOpp.prev = hedgeOpp.prev.prev;
                hedgeOpp.prev.next = hedgeOpp;
            }
            hedge.prev = hedgePrev.prev;
            hedge.prev.next = hedge;

            hedge.opposite = hedgeOpp;
            hedgeOpp.opposite = hedge;

            // oppFace was modified, so need to recompute
            oppFace.computeNormalAndCentroid();
        } else {
            hedgePrev.next = hedge;
            hedge.prev = hedgePrev;
        }
        return discardedFace;
    }

    // </editor-fold>
    
    // ==================================================================
    // <editor-fold defaultstate="collapsed" desc="Setter and Getter">
    /**
     * Gets the i-th half-edge associated with the face.
     *
     * @param idx the half-edge index, in the range 0-2.
     * @return the half-edge
     */
    public HalfEdge getEdge(int idx) {
        HalfEdge he = he0;
        while (idx > 0) {
            he = he.next;
            idx--;
        }
        while (idx < 0) {
            he = he.prev;
            idx++;
        }
        return he;
    }

    /**
     *
     * @return
     */
    public HalfEdge getFirstEdge() {
        return he0;
    }

    /**
     * Returns the normal of the plane associated with this face.
     *
     * @return the planar normal
     */
    public Vector3D getNormal() {
        return normal;
    }

    /**
     *
     * @return
     */
    public Point3D getCentroid() {
        return centroid;
    }

    /**
     *
     * @return
     */
    public int numVertices() {
        return numVerts;
    }

    /**
     *
     * @return
     */
    public double getArea() {
        return area;
    }

    /**
     *
     * @return
     */
    public STATUS getMark() {
        return mark;
    }

    /**
     *
     * @param mark
     */
    public void setMark(STATUS mark) {
        this.mark = mark;
    }

    /**
     *
     * @return
     */
    public Vertex getOutside() {
        return outside;
    }

    /**
     *
     * @param outside
     */
    public void setOutside(Vertex outside) {
        this.outside = outside;
    }

    /**
     *
     * @return
     */
    public double getPlaneOffset() {
        return planeOffset;
    }

    /**
     *
     * @param planeOffset
     */
    public void setPlaneOffset(double planeOffset) {
        this.planeOffset = planeOffset;
    }

    /**
     *
     * @return
     */
    public int getNumVerts() {
        return numVerts;
    }

    /**
     *
     * @param numVerts
     */
    public void setNumVerts(int numVerts) {
        this.numVerts = numVerts;
    }

    /**
     *
     * @return
     */
    public HullFace getNext() {
        return next;
    }

    /**
     *
     * @param next
     */
    public void setNext(HullFace next) {
        this.next = next;
    }

    /**
     *
     * @return
     */
    public String getVertexString() {
        String s = null;
        HalfEdge he = he0;

        do {
            if (s == null) {
                s = "" + he.getHead().index;
            } else {
                s += " " + he.getHead().index;
            }
            he = he.next;
        } while (he != he0);

        return s;
    }

    /**
     *
     * @param idxs
     */
    public void getVertexIndices(int[] idxs) {
        HalfEdge he = he0;
        int idx = 0;

        do {
            idxs[idx++] = he.getHead().index;
            he = he.next;
        } while (he != he0);
    }
    // </editor-fold>

    // ==================================================================
    /**
     * 
     */
    public void checkConsistency() {
        // do a sanity check on the face
        HalfEdge hedge = he0;
        double maxd = 0;
        int numv = 0;

        if (numVerts < 3) {
            throw new ArithmeticException(
                    "degenerate face: " + getVertexString());
        }

        do {
            HalfEdge hedgeOpp = hedge.getOpposite();
            if (hedgeOpp == null) {
                throw new ArithmeticException(
                        "face " + getVertexString() + ": "
                        + "unreflected half edge " + hedge.getVertexString());
            } else if (hedgeOpp.getOpposite() != hedge) {
                throw new ArithmeticException(
                        "face " + getVertexString() + ": "
                        + "opposite half edge " + hedgeOpp.getVertexString()
                        + " has opposite "
                        + hedgeOpp.getOpposite().getVertexString());
            }
            if (hedgeOpp.getHead() != hedge.getTail()
                    || hedge.getHead() != hedgeOpp.getTail()) {
                throw new ArithmeticException(
                        "face " + getVertexString() + ": "
                        + "half edge " + hedge.getVertexString()
                        + " reflected by " + hedgeOpp.getVertexString());
            }

            HullFace oppFace = hedgeOpp.face;
            if (oppFace == null) {
                throw new ArithmeticException(
                        "face " + getVertexString() + ": "
                        + "no face on half edge " + hedgeOpp.getVertexString());
            } else if (oppFace.mark == STATUS.DELETED) {
                throw new ArithmeticException(
                        "face " + getVertexString() + ": "
                        + "opposite face " + oppFace.getVertexString()
                        + " not on hull");
            }

            double d = Math.abs(distanceToPlane(hedge.getHead()));
            if (d > maxd) {
                maxd = d;
            }
            numv++;
            hedge = hedge.next;
        } while (hedge != he0);

        if (numv != numVerts) {
            throw new ArithmeticException(
                    "face " + getVertexString()
                    + " numVerts=" + numVerts + " should be " + numv);
        }

    }

    /**
     * Computes the distance from a point p to the plane of this face.
     *
     * @param p the point
     * @return distance from the point to the plane
     */
    public double distanceToPlane(Point3D p) {
        return normal.dotProduct(p) - planeOffset;
    }
}
