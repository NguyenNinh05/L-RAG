import { test, expect } from '@playwright/test'

test.describe('LegalDiff smoke tests', () => {
  test('login page loads', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('h1')).toContainText('LegalDiff')
    await expect(page.locator('button[type="submit"]')).toBeVisible()
  })

  test('register page loads', async ({ page }) => {
    await page.goto('/register')
    await expect(page.locator('h1')).toContainText('LegalDiff')
    await expect(page.locator('button[type="submit"]')).toBeVisible()
  })

  test('protected routes redirect to login', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveURL(/\/login/)
  })

  test('page has correct lang attribute', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('html')).toHaveAttribute('lang', 'vi')
  })
})
