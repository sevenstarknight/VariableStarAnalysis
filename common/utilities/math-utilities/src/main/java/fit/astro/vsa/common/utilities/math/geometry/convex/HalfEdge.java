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

/**
 * Represents the half-edges that surround each face in a counter-clockwise
 * direction.
 *
 * @author John E. Lloyd, Fall 2004
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class HalfEdge {

    /**
     * The vertex associated with the head of this half-edge.
     */
    public Vertex vertex;

    /**
     * Triangular face associated with this half-edge.
     */
    public HullFace face;

    /**
     * Next half-edge in the triangle.
     */
    public HalfEdge next;

    /**
     * Previous half-edge in the triangle.
     */
    public HalfEdge prev;

    /**
     * Half-edge associated with the opposite triangle adjacent to this edge.
     */
    public HalfEdge opposite;

    /**
     * Constructs a HalfEdge with head vertex <code>v</code> and left-hand
     * triangular face <code>f</code>.
     *
     * @param v head vertex
     * @param f left-hand triangular face
     */
    public HalfEdge(Vertex v, HullFace f) {
        vertex = v;
        face = f;
    }

    /**
     * Empty constructor for half-edge
     */
    public HalfEdge() {
    }

    // <editor-fold defaultstate="collapsed" desc="Getter and Setter">
    /**
     * Sets the value of the next edge adjacent (counter-clockwise) to this one
     * within the triangle.
     *
     * @param edge next adjacent edge
     */
    public void setNext(HalfEdge edge) {
        next = edge;
    }

    /**
     * Gets the value of the next edge adjacent (counter-clockwise) to this one
     * within the triangle.
     *
     * @return next adjacent edge
     */
    public HalfEdge getNext() {
        return next;
    }

    /**
     * Sets the value of the previous edge adjacent (clockwise) to this one
     * within the triangle.
     *
     * @param edge previous adjacent edge
     */
    public void setPrev(HalfEdge edge) {
        prev = edge;
    }

    /**
     * Gets the value of the previous edge adjacent (clockwise) to this one
     * within the triangle.
     *
     * @return previous adjacent edge
     */
    public HalfEdge getPrev() {
        return prev;
    }

    /**
     * Returns the triangular face located to the left of this half-edge.
     *
     * @return left-hand triangular face
     */
    public HullFace getHullFace() {
        return face;
    }

    /**
     * Returns the half-edge opposite to this half-edge.
     *
     * @return opposite half-edge
     */
    public HalfEdge getOpposite() {
        return opposite;
    }

    /**
     * Sets the half-edge opposite to this half-edge.
     *
     * @param edge opposite half-edge
     */
    public void setOpposite(HalfEdge edge) {
        opposite = edge;
        edge.opposite = this;
    }

    /**
     * Returns the head vertex associated with this half-edge.
     *
     * @return head vertex
     */
    public Vertex getHead() {
        return vertex;
    }

    /**
     * Returns the tail vertex associated with this half-edge.
     *
     * @return tail vertex
     */
    public Vertex getTail() {
        return prev != null ? prev.vertex : null;
    }

    /**
     * Returns the opposite triangular face associated with this half-edge.
     *
     * @return opposite triangular face
     */
    public HullFace getOppositeHullFace() {
        return opposite != null ? opposite.face : null;
    }

    /**
     * Produces a string identifying this half-edge by the point index values of
     * its tail and head vertices.
     *
     * @return identifying string
     */
    public String getVertexString() {
        if (getTail() != null) {
            return ""
                    + getTail().index + "-"
                    + getHead().index;
        } else {
            return "?-" + getHead().index;
        }
    }

    // </editor-fold>
    /**
     * Returns the length of this half-edge.
     *
     * @return half-edge length
     */
    public double length() {
        if (getTail() != null) {
            return getHead().distance(getTail());
        } else {
            return -1;
        }
    }

    /**
     * Returns the length squared of this half-edge.
     *
     * @return half-edge length squared
     */
    public double lengthSquared() {
        if (getTail() != null) {
            return getHead().distanceSq(getTail());
        } else {
            return -1;
        }
    }

    /**
     * Returns true if this vertex is empty.
     *
     * @return
     */
    public boolean isEmpty() {
        return vertex == null;
    }

}
