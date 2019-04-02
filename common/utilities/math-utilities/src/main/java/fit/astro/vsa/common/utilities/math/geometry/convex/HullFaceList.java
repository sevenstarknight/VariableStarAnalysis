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
 * Maintains a list of faces for use by QuickHull3D
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class HullFaceList {

    private HullFace head;
    private HullFace tail;

    /**
     * Clears this list.
     */
    public void clear() {
        head = tail = null;
    }

    /**
     * Adds a vertex to the end of this list.
     * @param vtx
     */
    public void add(HullFace vtx) {
        if (head == null) {
            head = vtx;
        } else {
            tail.setNext(vtx);
        }
        vtx.setNext(null);
        tail = vtx;
    }

    /**
     * Beginning of the Face (head)
     * @return 
     */
    public HullFace first() {
        return head;
    }

    /**
     * Returns true if this list is empty.
     * @return 
     */
    public boolean isEmpty() {
        return head == null;
    }
}
