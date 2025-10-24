/*
 Navicat Premium Data Transfer

 Source Server         : Localhost1
 Source Server Type    : MySQL
 Source Server Version : 50715
 Source Host           : localhost:3306
 Source Schema         : data_bank

 Target Server Type    : MySQL
 Target Server Version : 50715
 File Encoding         : 65001

 Date: 23/10/2025 18:24:36
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for metrics
-- ----------------------------
DROP TABLE IF EXISTS `metrics`;
CREATE TABLE `metrics`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'id del modelo',
  `accuracy` float NOT NULL COMMENT 'accuracy',
  `precision` float NOT NULL COMMENT 'precision',
  `recall` float NOT NULL COMMENT 'recall',
  `f1` float NOT NULL COMMENT 'f1',
  `roc_auc` float NOT NULL COMMENT 'roc',
  `confusion_matrix` json NOT NULL COMMENT 'matriz',
  `created_at` timestamp NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT 'registrar',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of metrics
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;
