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
package fit.astro.vsa.common.utilities.math.support;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class CollectionOperations {

    public static <T extends Comparable<T>> int findMedianIndex(final List<T> xs) {

        Collections.sort(xs);
        return xs.size() / 2;

    }

    /**
     *
     * @param <T>
     * @param xs
     * <p>
     * @return
     */
    public static <T extends Comparable<T>> int findMinIndex(final List<T> xs) {
        int minIndex;
        if (xs.isEmpty()) {
            minIndex = -1;
        } else {
            final ListIterator<T> itr = xs.listIterator();
            T min = itr.next(); // first element as the current minimum
            minIndex = itr.previousIndex();
            while (itr.hasNext()) {
                final T curr = itr.next();
                if (curr.compareTo(min) < 0) {
                    min = curr;
                    minIndex = itr.previousIndex();
                }
            }
        }
        return minIndex;
    }

    /**
     *
     * @param <T>
     * @param xs
     * <p>
     * @return
     */
    public static <T extends Comparable<T>> int findMaxIndex(final List<T> xs) {
        int maxIndex;
        if (xs.isEmpty()) {
            maxIndex = -1;
        } else {
            final ListIterator<T> itr = xs.listIterator();
            T max = itr.next(); // first element as the current minimum
            maxIndex = itr.previousIndex();
            while (itr.hasNext()) {
                final T curr = itr.next();
                if (curr.compareTo(max) > 0) {
                    max = curr;
                    maxIndex = itr.previousIndex();
                }
            }
        }
        return maxIndex;
    }

    /**
     *
     * @param labelTraining
     * <p>
     * @return
     */
    public static Set<String> generateUniqueClasses(
            String[] labelTraining) {
        Map<String, Integer> uniques = new HashMap<>();
        for (String a : labelTraining) {
            if (uniques.containsKey(a)) {
                uniques.put(a, uniques.get(a) + 1);
            } else {
                uniques.put(a, 1);
            }

        }

        return uniques.keySet();
    }

    /**
     *
     * @param labelTraining
     * <p>
     * @return
     */
    public static Set<String> generateUniqueClasses(
            Collection<String> labelTraining) {
        Map<String, Integer> uniques = new HashMap<>();
        for (String a : labelTraining) {
            if (uniques.containsKey(a)) {
                uniques.put(a, uniques.get(a) + 1);
            } else {
                uniques.put(a, 1);
            }

        }

        return uniques.keySet();
    }

    /**
     *
     * @param labelTraining
     * <p>
     * @return
     */
    public static Map<String, Integer> countClassTypes(
            Collection<String> labelTraining) {
        Map<String, Integer> uniques = new HashMap<>();
        for (String a : labelTraining) {
            if (uniques.containsKey(a)) {
                uniques.put(a, uniques.get(a) + 1);
            } else {
                uniques.put(a, 1);
            }

        }

        return uniques;
    }

}